"""Gauntlet eval env: robot launches into gap_0 of the progressive track and
runs as far as it can (gaps grow, platforms shrink) before falling.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

import src.tasks.parkour.mdp as mdp
from src.isaacs_go2.gauntlet_terrain import GAUNTLET_TERRAINS_CFG
from src.tasks.go2_safety_filter.gap.env_cfg import unitree_go2_gap_reach_avoid_env_cfg


def reset_gauntlet_start(env, env_ids, asset_cfg=SceneEntityCfg("robot"), start_x=-0.15):
  """Spawn at ``start_x`` (relative to gap_0's near edge) with a forward launch.

  ``start_x`` is negative = behind the gap on the approach platform; move it more
  negative to show the full takeoff arc before the first gap.
  """
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

  # Low noise so the launch is reliable (this is an eval terrain).
  pose = torch.stack(
    [start_x + u(-0.02, 0.02), u(-0.03, 0.03), 0.05 + u(-0.02, 0.02),
     u(-0.02, 0.02), u(-0.03, 0.03), u(-0.03, 0.03)], dim=1)
  vel = torch.stack(
    [2.6 + u(-0.05, 0.05), u(-0.03, 0.03), 0.55 + u(-0.05, 0.05),
     u(-0.05, 0.05), u(-0.05, 0.05), u(-0.05, 0.05)], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(root[:, 7:13] + vel, env_ids=env_ids)


def unitree_go2_gauntlet_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(GAUNTLET_TERRAINS_CFG)
  cfg.episode_length_s = 15.0  # long enough to run the whole track if it chains
  cfg.events["reset_base"] = EventTermCfg(func=reset_gauntlet_start, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  # Terminate the instant the base drops below platform level, so a failed gap
  # ends cleanly at the gap (no deep free-fall / physics blow-up corrupting the
  # distance metric).
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.base_too_low, params={"min_height": 0.15}
  )
  cfg.curriculum = {}
  return cfg
