"""Whole-body locomotion + manipulation environment for Go2 + Piper.

Replicates the ManipLoco (Fu/Cheng/Pathak, CoRL 2022) framework on our
Go2 + Piper hardware:

  * Action space is 18-D — both Go2 legs (12) and Piper arm (6) joints
    are actively controlled.
  * The policy is conditioned on TWO command channels: a base-velocity
    command (vx, ωyaw) and an end-effector pose command (p, q) ∈ SE(3)
    sampled in spherical coords around the arm base.
  * Reward is the sum of a locomotion stream (vel tracking, alive bonus,
    leg energy²) and a manipulation stream (EE pose tracking, arm energy).

Caveat: this v1 uses a single MLP and a single summed advantage. The
paper's "Advantage Mixing" (two value heads, mixing coefficient β
ramped 0→1) is the next thing to implement once this trains end-to-end;
without it the policy is liable to fall into a local minimum where it
tracks the EE while standing still. See note in the README.
"""

from __future__ import annotations

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp as builtin_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  GO2_JOINT_REGEX,
  PIPER_JOINT_REGEX,
  get_go2_piper_wholebody_robot_cfg,
)
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.ee_pose_command import UniformEEPoseCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "flat"]

# NOTE: SceneEntityCfg.resolve() mutates the cfg object, so each manager
# term needs its own instance. These are factories, not constants.
def _go2_joints_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(GO2_JOINT_REGEX,))


def _piper_joints_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(PIPER_JOINT_REGEX,))


def _all_actuated_joints_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", joint_names=(GO2_JOINT_REGEX, PIPER_JOINT_REGEX)
  )


def _add_arm_obs_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  """Inject EE pose observations into actor and critic groups, and let the
  joint-state observations cover both legs and arm (the actor needs to
  see the piper joints since the policy now controls them)."""

  # Restrict joint_pos / joint_vel to the actuated joints (legs + arm).
  for group in ("actor", "critic"):
    obs_group = cfg.observations[group]
    for term_name in ("joint_pos", "joint_vel"):
      if term_name in obs_group.terms:
        obs_group.terms[term_name].params = {
          **obs_group.terms[term_name].params,
          "asset_cfg": _all_actuated_joints_cfg(),
        }

  # Add EE state + EE command to both actor and critic.
  for group in ("actor", "critic"):
    obs_group = cfg.observations[group]
    obs_group.terms["ee_pose_b"] = ObservationTermCfg(
      func=mdp.ee_pose_b,
      params={"ee_site_name": "gripper_site"},
    )
    obs_group.terms["ee_command_b"] = ObservationTermCfg(
      func=mdp.ee_command_b,
      params={"command_name": "ee_pose"},
    )


def _add_arm_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add EE-pose-tracking and energy rewards; rescope existing leg
  rewards to Go2 joints only (so adding piper joints doesn't dilute
  pose / stand_still terms)."""

  # ---- existing reward fixes: scope to Go2 leg joints ----
  for rew_name in ("pose", "stand_still", "joint_acc_l2", "joint_pos_limits"):
    if rew_name in cfg.rewards:
      cfg.rewards[rew_name].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(GO2_JOINT_REGEX,)
      )

  # ---- EE tracking ----
  # Weight bumped from 0.5 → 3.0 after first wholebody run (wandb d0qatrmu)
  # converged with EE-tracking reward stuck at ~8% of max while every other
  # term reached 50–80%. Per-step locomotion-related reward summed to ~3.0
  # vs ~0.04 from EE tracking, so the policy correctly identified that
  # ignoring the arm was the high-reward strategy. Bigger weight makes the
  # EE gradient competitive with the locomotion gradient.
  cfg.rewards["track_ee_pose"] = RewardTermCfg(
    func=mdp.track_ee_pose_l1,
    weight=3.0,
    params={
      "command_name": "ee_pose",
      "std": 1.0,
      "ee_site_name": "gripper_site",
    },
  )

  # ---- Arm pose regularization ----
  # Keeps arm joints near their stowed default when not actively reaching.
  # Without this the arm has *no* gradient toward sensible postures (all
  # other regularization terms are scoped to GO2 leg joints), so under
  # ~0.5 noise std it just wiggles randomly. Low weight so the EE-tracking
  # reward can still pull the arm out of the default when needed.
  cfg.rewards["arm_pose"] = RewardTermCfg(
    func=builtin_mdp.variable_posture,
    weight=0.05,
    params={
      "asset_cfg": _piper_joints_cfg(),
      "command_name": "twist",
      "std_standing": {".*": 0.4},
      "std_walking": {".*": 0.4},
      "std_running": {".*": 0.4},
      "walking_threshold": 0.1,
      "running_threshold": 1.5,
    },
  )

  # ---- Energy split: arm (linear), legs (squared) ----
  cfg.rewards["arm_energy_l1"] = RewardTermCfg(
    func=mdp.joint_energy_l1,
    weight=-0.004,
    params={"asset_cfg": _piper_joints_cfg()},
  )
  cfg.rewards["leg_energy_sq"] = RewardTermCfg(
    func=mdp.joint_energy_sq,
    weight=-5e-5,
    params={"asset_cfg": _go2_joints_cfg()},
  )


def _add_dr_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Domain randomization following the paper's wider ranges. Reset
  events also randomize the piper joint positions so episodes start
  with the arm in varied poses."""

  # The default reset_robot_joints zeroes piper qpos because of clear_state;
  # add a separate reset that perturbs piper joints around their default.
  cfg.events["reset_arm_joints"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.5, 0.5),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": _piper_joints_cfg(),
    },
  )

  # ManipLoco-style env perturbations.
  cfg.events["randomize_base_payload_mass"] = EventTermCfg(
    func=dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("payload",)),
      "operation": "scale",
      "ranges": (0.5, 3.0),
    },
  )
  cfg.events["randomize_ee_payload_mass"] = EventTermCfg(
    func=dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("piper_gripper_link1",)),
      "operation": "add",
      "ranges": (0.0, 0.1),
    },
  )
  cfg.events["randomize_arm_mount_pos"] = EventTermCfg(
    func=dr.body_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("piper_mount",)),
      "operation": "add",
      "ranges": {0: (-0.05, 0.05), 1: (-0.03, 0.03), 2: (-0.02, 0.02)},
    },
  )


def unitree_go2_piper_wholebody_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Whole-body Go2 + Piper, rough terrain."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_piper_wholebody_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # ------------------------------------------------------------------ actions
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # Whole-body: control all 18 joints (legs + piper arm).
  joint_pos_action.actuator_names = (GO2_JOINT_REGEX, PIPER_JOINT_REGEX)
  # Per-group action scales: legs ~0.25 (existing), arm at the paper's
  # delta range divided by ~2 to keep RL output in a reasonable [-1, 1].
  joint_pos_action.scale = {
    GO2_JOINT_REGEX: 0.25,
    PIPER_JOINT_REGEX: 0.5,
  }

  # ------------------------------------------------------------------ commands
  cfg.commands["ee_pose"] = UniformEEPoseCommandCfg(
    entity_name="robot",
    anchor_body_name="piper_mount",
    ee_site_name="gripper_site",
    resampling_time_range=(1.0, 3.0),
    debug_vis=True,
    ranges=UniformEEPoseCommandCfg.Ranges(
      l=(0.2, 0.6),
      pitch=(-2 * math.pi / 5, 2 * math.pi / 5),
      yaw=(-3 * math.pi / 5, 3 * math.pi / 5),
    ),
  )
  # Loosen the existing twist command to ManipLoco's forward-only band.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.9)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.05

  # ------------------------------------------------------------------ obs/rewards
  _add_arm_obs_terms(cfg)
  _add_arm_rewards(cfg)
  _add_dr_events(cfg)

  # Existing per-Go2 wiring (foot sites, base body, foot friction geoms).
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=builtin_mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # ------------------------------------------------------------------ misc
  # Limit reset joints event to legs (piper has its own reset_arm_joints).
  if "reset_robot_joints" in cfg.events:
    cfg.events["reset_robot_joints"].params["asset_cfg"] = _go2_joints_cfg()

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_piper_wholebody_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Whole-body Go2 + Piper, flat ground variant."""
  cfg = unitree_go2_piper_wholebody_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  if "height_scan" in cfg.observations["actor"].terms:
    del cfg.observations["actor"].terms["height_scan"]
  if "height_scan" in cfg.observations["critic"].terms:
    del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  return cfg
