"""Walking environment config for the Go2 safety shield stack.

This is the **nominal** locomotion policy — trained on flat + rough terrain
(no gaps, no crawl barriers, per user spec) with the same proprioception
observation group used by the safety task, so both policies can be combined
at play time without an observation adapter.

"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from src.assets.robots import get_go2_robot_cfg
import src.tasks.parkour.mdp as mdp
from src.tasks.parkour.parkour_env_cfg import make_parkour_env_cfg


def unitree_go2_walking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 walking environment on rough terrain (no gaps/crawl)."""
  cfg = make_parkour_env_cfg()

  # --- Sim tuning (simpler terrain, smaller contact budget) ---
  cfg.sim.nconmax = 80
  cfg.sim.njmax = 800
  cfg.sim.mujoco.ccd_iterations = 200
  cfg.sim.contact_sensor_maxmatch = 200

  # --- Robot ---
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # --- Terrain: rough (no gaps, no crawl) ---
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(ROUGH_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 3

  # --- Sensors: drop the depth camera, keep raycast terrain scan + contacts ---
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = foot_names
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  kept_sensors = []
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "front_depth":
      # Walking policy is blind to vision.
      continue
    if isinstance(sensor, RayCastSensorCfg) and sensor.name == "terrain_scan":
      sensor.frame.name = "base_link"
    kept_sensors.append(sensor)

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
  cfg.scene.sensors = tuple(kept_sensors) + (feet_ground_cfg, nonfoot_ground_cfg)

  # --- Observations: remove the depth group (MLP actor). ---
  cfg.observations.pop("depth", None)

  # --- Actions: inherit JointPositionActionCfg from parkour factory ---
  # (scale 0.25, ".*" actuators) — already correct for Go2.

  # --- Commands: full locomotion range with heading control. ---
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.resampling_time_range = (3.0, 8.0)
  twist.rel_standing_envs = 0.05
  twist.rel_heading_envs = 0.25
  twist.heading_command = True
  twist.heading_control_stiffness = 0.5
  twist.ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 2.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    heading=(-3.14159, 3.14159),
  )

  # --- Events ---
  # Use the standard on-ground reset (drop the parkour mid-air spawn).
  # `reset_base` already does this in the parent factory — no change needed.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # --- Rewards: strip parkour-specific terms ---
  cfg.rewards.pop("gap_crossing", None)
  cfg.rewards.pop("gap_crossing_bonus", None)
  cfg.rewards.pop("forward_progress", None)

  # Per-robot reward params (same Go2 wiring as parkour env).
  cfg.rewards["body_orientation"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_height"].params["target_height"] = 0.30
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["feet_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]  # Trot.
  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names

  # --- Terminations: drop base_too_low (no gaps on this terrain). ---
  cfg.terminations.pop("base_too_low", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # --- Per-robot observation wiring (critic group only — actor has no site list). ---
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  # --- Curriculum: distance-based terrain level up/down. ---
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=velocity_mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
  }

  # --- Viewer ---
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  # --- Episode length ---
  cfg.episode_length_s = 20.0

  # --- Play mode overrides ---
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["proprioception"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    assert cfg.scene.terrain is not None
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
