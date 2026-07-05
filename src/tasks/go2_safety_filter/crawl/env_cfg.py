"""Crawl-filter task: keep moving forward under a bar, ducking as low as it
takes — or STOP if the bar is below the crouch feasibility floor.

Strategy (v2, height curriculum — see chat 2026-07-04): the crawl skill is
acquired CONTINUOUSLY, never as a rare exploration win.  The terrain starts
with a bar high enough that the standing/walking Go2 passes untouched
(row 0 = 0.50 m clearance), and a forward curriculum lowers it one small notch
at a time; each notch demands only a slightly deeper duck than the last, so
ducking emerges as a smooth extension of walking.  Below the feasibility floor
(~0.22 m) the bar is impossible and the correct behavior is to stop.

Reach-avoid objective (COMMAND / velocity-liveness mode, NOT the jumping
line's rest mode):
    g (avoid)  = min(base-height, orientation, no-nonfoot-contact)  — don't
                 fall / tip / strike the bar
    l (reach)  = forward-speed liveness (>= a fraction of the commanded vx)
Under a passable bar the robot keeps forward speed by ducking (l>=0, g>=0 ->
V>0); under an impossible one, keeping speed means striking, so the only way
to hold g>=0 is to stop -> l<0 -> V<0.  The value function is negative exactly
on the impassable bars — the stop-vs-go signal the deployment filter reads.
(Rest mode collapsed to stop-always here: the Go2 can brake out of almost any
approach, so "reach safe rest" was satisfiable everywhere.)

Deployment: the crawl policy is the safety backup while a bar is overhead;
once the robot is clear, the nominal task policy takes over.

Perception: two forward raycast fans (bar_scan up ~17deg, bar_scan_low near
horizontal) on the actor+critic; privileged analytic [dist, clearance] on the
critic only. The down-looking terrain_scan is blind to overhead bars.
"""

from __future__ import annotations

import copy

import torch
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

import src.tasks.parkour.mdp as mdp
from src.isaacs_go2.crawl_filter_terrain import (
  BAR_DEPTH,
  CRAWL_FILTER_TERRAINS_CFG,
  _BAR_X,
)

# Spawn geometry.
_NOSE = 0.35        # body reach ahead of base center
_APPROACH_MIN = 0.3  # nose-distance to the bar face, near end
_APPROACH_MAX = 2.2  # nose-distance to the bar face, far end (approach is 2.5)


def reset_crawl_approach(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn upright on the approach, facing the bar, with forward momentum.

  Distance-to-bar and arrival speed are both varied so the value function sees
  the full decision-relevant band (far/slow trivially safe -> close/fast
  possibly doomed at low bars).  No crouch, no strata: the terrain height
  curriculum, not the spawn, sets the difficulty."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  d = u(_APPROACH_MIN, _APPROACH_MAX)                 # nose-distance to bar face
  x = (_BAR_X - _NOSE - d).clamp(min=0.15)
  pose = torch.stack(
    [x, u(-0.06, 0.06), u(-0.02, 0.02),
     u(-0.05, 0.05), u(-0.05, 0.05), u(-0.06, 0.06)], dim=1)
  vx = u(0.3, 1.8)
  velocities = root[:, 7:13] + torch.stack(
    [vx, u(-0.10, 0.10), u(-0.05, 0.05),
     u(-0.10, 0.10), u(-0.10, 0.10), u(-0.10, 0.10)], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def crawl_height_levels(env, env_ids) -> torch.Tensor | None:
  """Forward height curriculum, bar-relative (robust to the varied spawn x):
  promote when the robot got THROUGH the bar and kept going (>1 m past exit),
  demote when it ended short of the bar (stopped / never reached it).  Read
  at reset time, before respawn, so the position is the episode's final one.

  On impossible rows the robot stops before the bar -> demote, so the
  population settles at the feasibility frontier and the impossible rows are
  visited transiently (enough for V<0 to form there)."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  x_rel = (env.scene["robot"].data.root_link_pos_w[env_ids, 0]
           - env.scene.env_origins[env_ids, 0])
  # Guard the first reset (robot world position is stale/off-patch before the
  # reset event places it -> x_rel ~ inter-patch spacing): only trust readings
  # inside the 12 m patch.
  in_patch = (x_rel > -1.0) & (x_rel < 11.0)
  move_up = in_patch & (x_rel > (_BAR_X + BAR_DEPTH + 1.0))
  move_down = in_patch & (x_rel < (_BAR_X - 0.3))
  terrain.update_env_origins(env_ids, move_up, move_down & ~move_up)
  return terrain.terrain_levels.float().mean()


def pinned_levels_crawl(env, env_ids) -> torch.Tensor:
  """Adversarial phases: pin the terrain rows the ctrl policy mastered so the
  survival-gated curriculum can't demote under adversarial pressure."""
  return env.scene.terrain.terrain_levels.float().mean()


# --- perception ---------------------------------------------------------------

def _add_bar_perception(cfg: ManagerBasedRlEnvCfg) -> None:
  """Forward/up raycast fans (actor+critic) + analytic bar info (critic)."""
  bar_scan = RayCastSensorCfg(
    name="bar_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.0, 0.6), resolution=0.1,
                           direction=(1.0, 0.0, 0.3)),
    max_distance=4.0,
    exclude_parent_body=True,
  )
  bar_scan_low = RayCastSensorCfg(
    name="bar_scan_low",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.0, 0.6), resolution=0.1,
                           direction=(1.0, 0.0, 0.05)),
    max_distance=4.0,
    exclude_parent_body=True,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (bar_scan, bar_scan_low)
  for name in ("bar_scan", "bar_scan_low"):
    cfg.observations["proprioception"].terms[name] = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0})
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0})
  cfg.observations["critic"].terms["bar_info"] = ObservationTermCfg(
    func=mdp.bar_info, params={})


# --- env cfg builders ---------------------------------------------------------

def unitree_go2_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from src.tasks.go2_safety_filter.gap.env_cfg import (
    unitree_go2_gap_reach_avoid_env_cfg,
  )

  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CRAWL_FILTER_TERRAINS_CFG)
  # Forward curriculum: every env starts at the highest (trivial) bar.
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.episode_length_s = 6.0

  # Side-profile follow camera (azimuth 90 = robot moving L->R into the bar on
  # the right): the crouch depth is directly legible in profile. The bar spans
  # the full width, so a side view looks THROUGH it when the robot is under —
  # only works because the beam/wall are now semi-transparent (verified: robot
  # stays fully visible under the bar). A rear view (azimuth 180) renders the
  # robot off-frame; azimuth 0/90/270 all show it, 90 gives natural motion.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 3.3
  cfg.viewer.elevation = -14.0
  cfg.viewer.azimuth = 90.0

  _add_bar_perception(cfg)

  # Ground approach spawn with forward momentum (no midair gap reset).
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_crawl_approach, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  # The parkour base re-rolls terrain each reset in play mode; it breaks the
  # height curriculum (and row pinning in evals) — drop it.
  cfg.events.pop("randomize_terrain", None)

  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=crawl_height_levels),
  }
  return cfg


def unitree_go2_crawl_isaacs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crawl_env_cfg(play=play)
  # Adversarial phases pin the mastered rows (no promote/demote treadmill).
  cfg.curriculum = {"pinned_levels": CurriculumTermCfg(func=pinned_levels_crawl)}
  return cfg
