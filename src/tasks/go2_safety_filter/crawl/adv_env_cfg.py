"""Crawl + RANDOM-PUSH curriculum (ISAACS increment 0 for the bar task).

Identical to the crawl task, plus a sustained random base push each episode
(random unit direction x level*5 N; per-env level curriculum: promote on
surviving to rest, demote on falling).  Pre-adversary hardening + the RANDOM
column of the crawl ISAACS matrix.  Reuses ``set_random_push``/``push_levels``
from the crossing-chain adv cfg verbatim (they only touch the wrench channel
and the timeout signal — nothing terrain-specific).

The learned-adversary task registers the PLAIN crawl ISAACS cfg — the wrench
channel must have exactly one owner.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg

from src.tasks.go2_safety_filter.crossing_chain.adv_env_cfg import (
  push_levels,
  set_random_push,
)
from src.tasks.go2_safety_filter.crawl.env_cfg import unitree_go2_crawl_env_cfg


def unitree_go2_crawl_adv_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crawl_env_cfg(play=play)
  # Appended last (after crouch/handover joints + obstacle window): wrench-only.
  cfg.events["set_random_push"] = EventTermCfg(
    func=set_random_push, mode="reset", params={}
  )
  cfg.curriculum["push_levels"] = CurriculumTermCfg(func=push_levels)
  return cfg
