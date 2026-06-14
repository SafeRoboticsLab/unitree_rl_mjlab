"""Go2 + Piper arm + onboard payload velocity environment configuration.

Same observation and action space as the Go2-only velocity env — the
arm + payload act as a heavy static load mounted on the robot's back.
Arm pose, arm mount position, payload position, and payload mass are
domain-randomized at episode start so the learned walking policy must
cope with a distribution of heavy back-loads.
"""

from typing import Literal

from src.assets.robots import (
  get_go2_piper_robot_cfg,
  GO2_JOINT_REGEX,
  PIPER_JOINT_REGEX,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.mdp.payload_events import reset_static_arm_pose
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

# Scene-entity configs restricting manager terms to the Go2 leg joints only.
_GO2_JOINTS_CFG = SceneEntityCfg("robot", joint_names=(GO2_JOINT_REGEX,))


def _restrict_to_go2_joints(cfg: ManagerBasedRlEnvCfg) -> None:
  """Scope obs / rewards / events to Go2 leg joints so the RL problem is
  identical to the Go2-only walking env. Piper joints still exist in the
  physics entity (held by their XML-defined actuators) but are invisible
  to the policy and critic."""

  # Observations.
  for group in ("actor", "critic"):
    obs_group = cfg.observations[group]
    for term_name in ("joint_pos", "joint_vel"):
      term = obs_group.terms[term_name]
      term.params = {**term.params, "asset_cfg": _GO2_JOINTS_CFG}

  # Actions: control only Go2 actuators via the RL policy.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (GO2_JOINT_REGEX,)

  # Rewards that enumerate joints.
  for rew_name in ("pose", "stand_still"):
    if rew_name in cfg.rewards:
      cfg.rewards[rew_name].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(GO2_JOINT_REGEX,)
      )
  # Rewards that use the default _DEFAULT_ASSET_CFG (all joints) — restrict.
  for rew_name in ("joint_acc_l2", "joint_pos_limits"):
    if rew_name in cfg.rewards:
      cfg.rewards[rew_name].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(GO2_JOINT_REGEX,)
      )

  # Events: reset_robot_joints should only randomize Go2 joints (the piper
  # arm gets its own dedicated randomization event below).
  if "reset_robot_joints" in cfg.events:
    cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(GO2_JOINT_REGEX,)
    )


def _add_arm_and_payload_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Domain-randomization events for the piper arm + payload.

  Four axes of variation, all applied at startup / reset:
    1. Arm pose — piper joint positions reset to a random offset around
       the stowed default, so the arm sits in a different configuration
       each episode (its actuators hold it there).
    2. Arm mount position — piper_mount body shifted around its default
       on the Go2 back (forward/back/left/right/up/down perturbation).
    3. Payload position — payload body shifted around its default spot
       behind the arm on the Go2 back.
    4. Payload mass — mass of the payload body scaled over a wide range
       (representing battery + compute of variable size).
  """

  # 1. Piper arm pose randomization — ±1.0 rad around stowed default per
  #    joint. We MUST write both qpos and joint_pos_target so the high-
  #    stiffness piper actuators hold the randomized pose instead of
  #    snapping the arm back to its default setpoint.
  cfg.events["randomize_arm_pose"] = EventTermCfg(
    func=reset_static_arm_pose,
    mode="reset",
    params={
      "position_range": (-1.0, 1.0),
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(PIPER_JOINT_REGEX,)
      ),
    },
  )

  # 2. Arm mount position — ±5cm front/back, ±3cm lateral, ±2cm vertical.
  cfg.events["randomize_arm_mount_pos"] = EventTermCfg(
    func=dr.body_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("piper_mount",)),
      "operation": "add",
      "ranges": {
        0: (-0.05, 0.05),
        1: (-0.03, 0.03),
        2: (-0.02, 0.02),
      },
    },
  )

  # 3. Payload position — slightly wider range behind the arm.
  cfg.events["randomize_payload_pos"] = EventTermCfg(
    func=dr.body_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("payload",)),
      "operation": "add",
      "ranges": {
        0: (-0.06, 0.04),
        1: (-0.04, 0.04),
        2: (-0.02, 0.03),
      },
    },
  )

  # 4. Payload mass — scale the base 2kg payload by 0.5x–2.5x
  #    (i.e. ~1kg–5kg, representing different battery + compute combos).
  cfg.events["randomize_payload_mass"] = EventTermCfg(
    func=dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("payload",)),
      "operation": "scale",
      "ranges": (0.5, 2.5),
    },
  )


def unitree_go2_piper_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 + Piper arm + payload rough terrain velocity config."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_piper_robot_cfg()}

  # Raycast sensor tracks base_link (same as Go2-only).
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
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Scope all joint-related manager terms to Go2 legs, then add the
  # piper + payload domain-randomization events.
  _restrict_to_go2_joints(cfg)
  _add_arm_and_payload_dr(cfg)

  # Apply play-mode overrides.
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


def unitree_go2_piper_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 + Piper flat terrain velocity configuration."""
  cfg = unitree_go2_piper_rough_env_cfg(play=play)

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
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
