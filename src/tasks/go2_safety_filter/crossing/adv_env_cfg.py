"""Crossing + adversary (increment 1): the launch-cross-land substrate must
survive a curriculum PUSH.

Spawns at the committed launch (the full crossing the substrate already learned)
and applies a sustained base force of random direction whose magnitude ramps via
a per-env curriculum: an env that still crosses safely gets a stronger push next
episode, one that fails gets a weaker one. Warm-start from the converged
crossing checkpoint; SafetyPPO (avoid-only g) then makes the crossing robust to
the push. (Increment 2 will replace the random push with a *learned* worst-case
disturbance = the ISAACS adversary proper.)
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from src.isaacs_go2.island_terrain import ISLAND_CROSSING_TERRAINS_CFG
from src.tasks.go2_safety_filter.gap.env_cfg import unitree_go2_gap_reach_avoid_env_cfg

FORCE_STEP = 5.0   # N per force level
MAX_FORCE_LEVEL = 12  # -> up to 60 N
_FAR_X = 0.7


def _ensure_force_level(env):
  if not hasattr(env, "_force_level"):
    env._force_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  if not hasattr(env, "_adv_body_ids"):
    ids, _ = env.scene["robot"].find_bodies("base_link")
    env._adv_body_ids = list(ids)


def reset_launch_with_push(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn at the committed launch and set a sustained random-direction push
  of curriculum magnitude for this episode."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_force_level(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  # Committed launch at the near edge (== back_level 5 of the crossing curriculum).
  pose = torch.stack(
    [-0.10 + u(-0.05, 0.05), u(-0.10, 0.10), 0.05 + u(-0.05, 0.05),
     u(-0.10, 0.10), u(-0.15, 0.15), u(-0.15, 0.15)], dim=1)
  vel = torch.stack(
    [2.5 + u(-0.20, 0.20), u(-0.10, 0.10), 0.6 + u(-0.20, 0.20),
     u(-0.2, 0.2), u(-0.2, 0.2), u(-0.2, 0.2)], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(root[:, 7:13] + vel, env_ids=env_ids)

  # Sustained push: random unit direction * (force_level * FORCE_STEP) N on base.
  mag = env._force_level[env_ids].float() * FORCE_STEP  # (n,)
  d = sample_uniform(-1.0, 1.0, (n, 3), device)
  d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
  forces = (d * mag[:, None]).reshape(n, 1, 3)
  asset.write_external_wrench_to_sim(
    forces, torch.zeros_like(forces), body_ids=env._adv_body_ids, env_ids=env_ids
  )


def push_force_levels(env, env_ids) -> torch.Tensor:
  """Ramp the push: promote force_level if the env still crossed safely
  (survived + on far platform upright), demote otherwise."""
  _ensure_force_level(env)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  up = -robot.data.projected_gravity_b[env_ids, 2]
  survived = env.termination_manager.time_outs[env_ids]
  reached = survived & (x_rel > _FAR_X) & (up > 0.7)
  lvl = env._force_level[env_ids]
  lvl = torch.where(reached, lvl + 1, lvl - 1)
  env._force_level[env_ids] = lvl.clamp(0, MAX_FORCE_LEVEL)
  return env._force_level.float().mean() * FORCE_STEP  # mean push (N)


def unitree_go2_crossing_adv_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(ISLAND_CROSSING_TERRAINS_CFG)
  cfg.events["reset_base"] = EventTermCfg(func=reset_launch_with_push, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if not play and "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  cfg.curriculum = {"push_levels": CurriculumTermCfg(func=push_force_levels)}
  return cfg
