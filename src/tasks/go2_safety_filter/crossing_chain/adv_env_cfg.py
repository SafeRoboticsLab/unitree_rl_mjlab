"""Crossing-chain + RANDOM-PUSH curriculum (ISAACS increment 0).

Identical to the rest-objective crossing-chain task, plus a sustained random
base push each episode: random unit direction x (level * 5 N), with a per-env
level curriculum (promote on surviving to rest, demote on falling).  This is
the pre-adversary baseline: it hardens the policy against *unintelligent*
disturbances and provides the RANDOM column of the ISAACS evaluation matrix.

The push is applied ONCE at reset via ``write_external_wrench_to_sim`` (the
wrench persists across steps until rewritten or sim reset), so no per-step
machinery is needed.  The event is appended AFTER ``handover_joints`` so it
only writes the wrench — root/joint state set by the takeover strata is
untouched.

Increment 1 (learned adversary) replaces this with a per-step force policy;
that task registers the PLAIN crossing-chain cfg — the wrench channel must
have exactly one owner.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.go2_safety_filter.crossing_chain.env_cfg import (
  unitree_go2_crossing_chain_env_cfg,
)

FORCE_STEP = 5.0        # N per curriculum level
MAX_PUSH_LEVEL = 10     # -> up to 50 N (the ISAACS game bound)


def _ensure_push_buffers(env):
  if not hasattr(env, "_push_level"):
    env._push_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._adv_body_ids = None


def set_random_push(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Write a sustained random-direction push on the base for this episode."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_push_buffers(env)
  asset = env.scene[asset_cfg.name]
  if env._adv_body_ids is None:
    env._adv_body_ids = asset.find_bodies("base_link")[0]
  n = int(len(env_ids))
  device = env.device

  direction = torch.randn(n, 3, device=device)
  direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
  magnitude = (env._push_level[env_ids].float() * FORCE_STEP).unsqueeze(-1)
  forces = (direction * magnitude).unsqueeze(1)  # (n, 1 body, 3)
  torques = torch.zeros_like(forces)
  asset.write_external_wrench_to_sim(
    forces, torques, body_ids=env._adv_body_ids, env_ids=env_ids
  )


def push_levels(env, env_ids) -> torch.Tensor:
  """Per-env push curriculum: promote on surviving to rest (timeout), demote on
  falling — matches this task's reach-safe-rest objective (NOT the old crossing
  task's reached-far-platform test)."""
  _ensure_push_buffers(env)
  time_outs = env.termination_manager.time_outs[env_ids]
  lvl = env._push_level[env_ids]
  lvl = torch.where(time_outs, lvl + 1, lvl - 1)
  env._push_level[env_ids] = lvl.clamp(0, MAX_PUSH_LEVEL)
  return env._push_level.float().mean()


def unitree_go2_crossing_chain_adv_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crossing_chain_env_cfg(play=play)
  # Appended last (after handover_joints): wrench-only, state writes untouched.
  cfg.events["set_random_push"] = EventTermCfg(
    func=set_random_push, mode="reset", params={}
  )
  cfg.curriculum["push_levels"] = CurriculumTermCfg(func=push_levels)
  return cfg
