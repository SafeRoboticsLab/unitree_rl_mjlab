"""Crawl-filter task: arrive with momentum at a low bar — STOP if the bar is
impossibly low, otherwise LOWER the body and CRAWL through, then settle.

Rest objective (l = safe rest + mild crossing bias) with a bar-specific rest
window published to the reach-avoid wrapper via ``env._rest_obstacle_window_w``,
encoding the stop-vs-crawl dichotomy per row: on PASSABLE rows only rest PAST
the bar counts (crawl-through is the target — plain rest made braking a
universal solution and the policy converged to stop-always); on IMPOSSIBLE
rows rest before the bar is the target.

Decision physics (nose-distance d to the bar face, speed v, brake decel a=3.0,
crouch time t_c=0.45 s):
  brakeable:   v < sqrt(2 a (d - stop_margin))
  crawlable:   d > v t_c - 0.5 a t_c^2            (duck while braking)
  MUST-CRAWL:  unstoppable but crawlable — the stratum that forces the skill
  DOOMED:      d < v t_min - 0.5 a t_min^2 (t_min=0.25) — teaches V < 0

Perception: two forward raycast fans (bar_scan up ~17deg, bar_scan_low near
horizontal) on the actor+critic; privileged analytic [dist, clearance] on the
critic only. The down-looking terrain_scan cannot see overhead bars.
"""

from __future__ import annotations

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
  bar_clearance_for_level,
  is_impossible_level,
)
from src.tasks.go2_safety_filter.crossing_chain.env_cfg import (
  _handover_data,
  phase_random_offset,
)

# --- decision-physics constants ---------------------------------------------
_A_BRAKE = 3.0
_T_CROUCH = 0.45
_T_MIN = 0.25
_NOSE = 0.35          # body reach ahead of base center
_VX_CAP = 3.4

# Spawn strata (passable rows).
_FRAC_STOPPABLE = 0.30
_FRAC_MUSTCRAWL = 0.30
_FRAC_MIDCRAWL = 0.15
_FRAC_DOOMED = 0.05   # remainder 0.20 = handover replay
# Impossible rows re-weight: stoppable 0.55 / doomed 0.25 / handover 0.20.

_CRAWL_LEVELS = 5

# Crouch joint poses (thigh, calf); hips stay ~default.
_CROUCH_SHALLOW = (1.2, -2.3)   # base ~0.19
_CROUCH_DEEP = (1.35, -2.55)    # base ~0.15 (calf soft-min ~-2.63)


def _ensure_crawl_buffers(env):
  if not hasattr(env, "_crawl_level"):
    n, dev = env.num_envs, env.device
    env._crawl_level = torch.zeros(n, dtype=torch.long, device=dev)
    env._was_mustcrawl = torch.zeros(n, dtype=torch.bool, device=dev)
    env._was_crawlwin = torch.zeros(n, dtype=torch.bool, device=dev)
    env._crouch_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._crouch_alpha = torch.zeros(n, device=dev)
    env._handover_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._handover_jpos = torch.zeros(n, 12, device=dev)
    env._handover_jvel = torch.zeros(n, 12, device=dev)
    env._handover_level = torch.zeros(n, dtype=torch.long, device=dev)
    env._rest_obstacle_window_w = torch.zeros(n, 2, device=dev)


def set_rest_obstacle_window(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Publish the per-env rest-exclusion window (world x) encoding the task's
  stop-vs-crawl dichotomy:

  * PASSABLE rows: the whole approach is excluded — only rest PAST the bar
    counts as reached, so crawling through IS the target maneuver. (Plain
    rest-before-bar made braking a universal solution: real braking ~5 m/s^2
    beats the analytic 3.0, the must-crawl window never binds, and the policy
    converges to stop-always — indistinguishable from the -Avoid baseline.)
  * IMPOSSIBLE rows: rest before the bar is the target (stop!); past-bar rest
    is unreachable anyway.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  terrain = env.scene.terrain
  impossible = is_impossible_level(terrain.terrain_levels[env_ids])
  ox = env.scene.env_origins[env_ids, 0]
  lo_impossible = ox + _BAR_X - 0.35
  lo_passable = ox - 100.0
  env._rest_obstacle_window_w[env_ids, 0] = torch.where(
    impossible, lo_impossible, lo_passable)
  env._rest_obstacle_window_w[env_ids, 1] = ox + _BAR_X + BAR_DEPTH + 0.40


def reset_takeover_crawl(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                         stop_margin: float = 0.0):
  """Stratified spawn across the stop-vs-crawl decision boundary."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  # Benchmark hook: `env._eval_spawn = {"d": ..., "v": ...}` overrides the
  # strata with a controlled upright spawn (nose-distance d, speed v) so eval
  # scripts stay in-band (out-of-band teleports leave staged writes + stale
  # raycast caches that misfire terrain-relative g/terminations for a step).
  ev = getattr(env, "_eval_spawn", None)
  if ev is not None:
    env._was_mustcrawl[env_ids] = False
    env._handover_mask[env_ids] = False
    env._crouch_mask[env_ids] = False
    positions = root[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0] += _BAR_X - _NOSE - float(ev["d"])
    positions[:, 2] += 0.02
    velocities = root[:, 7:13].clone()
    velocities[:] = 0.0
    velocities[:, 0] = float(ev["v"])
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, root[:, 3:7]], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
    return

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  terrain = env.scene.terrain
  levels = terrain.terrain_levels[env_ids]
  clearance = bar_clearance_for_level(levels)
  impossible = is_impossible_level(levels)
  hdata = _handover_data(device)

  r = u(0.0, 1.0)
  # Passable-row strata boundaries.
  stoppable = r < _FRAC_STOPPABLE
  mustcrawl = (r >= _FRAC_STOPPABLE) & (r < _FRAC_STOPPABLE + _FRAC_MUSTCRAWL)
  midcrawl = (r >= 0.60) & (r < 0.60 + _FRAC_MIDCRAWL)
  doomed = (r >= 0.75) & (r < 0.75 + _FRAC_DOOMED)
  handover = r >= 0.80
  # Impossible rows: no crawl mass — stoppable 0.55 / doomed 0.25 / handover 0.20.
  stoppable = torch.where(impossible, r < 0.55, stoppable)
  doomed = torch.where(impossible, (r >= 0.55) & (r < 0.80), doomed)
  mustcrawl = mustcrawl & ~impossible
  midcrawl = midcrawl & ~impossible
  handover = torch.where(impossible, r >= 0.80, handover)
  if hdata is None:
    stoppable = stoppable | handover
    handover = torch.zeros_like(handover)

  env._was_mustcrawl[env_ids] = mustcrawl
  env._was_crawlwin[env_ids] = mustcrawl | midcrawl
  env._handover_mask[env_ids] = handover
  env._crouch_mask[env_ids] = False

  # --- distances / speeds per stratum (nose-distance d to the bar face) ----
  assist = (1.0 - env._crawl_level[env_ids].float() / _CRAWL_LEVELS).clamp(0, 1)

  # stoppable: room to brake before the (margin-shifted) bar face
  d = u(0.5, 2.2)
  v_brake = torch.sqrt(2.0 * _A_BRAKE * (d - stop_margin).clamp_min(0.05))
  vx = (v_brake * u(0.15, 0.90)).clamp(0.0, _VX_CAP)

  # must-crawl: pre-crouched near the bar at high assist, else upright
  # fast+close inside the crawlable cone
  pre_crouched = mustcrawl & (u(0.0, 1.0) < assist)
  d_mc = torch.where(
    pre_crouched,
    u(0.15, 0.45),
    u(0.55, 0.85) + (1.0 - assist) * u(0.0, 0.9),
  )
  v_unstop = torch.sqrt(2.0 * _A_BRAKE * d_mc)
  vx_mc = torch.maximum(u(1.6, 2.8), v_unstop * 1.1).clamp(0.8, _VX_CAP)
  vx_mc = torch.minimum(vx_mc, (d_mc + 0.304) / _T_CROUCH)  # crawlable cone
  vx_mc = torch.where(pre_crouched, u(0.8, 1.6), vx_mc)
  d = torch.where(mustcrawl, d_mc, d)
  vx = torch.where(mustcrawl, vx_mc, vx)

  # doomed: cannot stop, cannot get low in time
  vx_dm = u(2.6, _VX_CAP)
  d_dm_hi = (_T_MIN * vx_dm - 0.5 * _A_BRAKE * _T_MIN**2).clamp_min(0.08)
  d = torch.where(doomed, u(0.05, 1.0) * d_dm_hi, d)
  vx = torch.where(doomed, vx_dm, vx)

  # positions: x measured so the NOSE is d before the bar face
  x = (_BAR_X - _NOSE - d).clamp(min=0.15)
  z = 0.05 + u(-0.02, 0.02)
  vz = u(-0.05, 0.05)

  # mid-crawl: crouched, moving through/out of the bar. The easiest rung
  # spawns FULLY CLEARED of the beam (trunk extends +-0.35 m from the base, so
  # even a base at the exit face still has its rear under the beam and the
  # policy's stand-up reflex strikes): crouched past exit+0.35, the win is
  # literally "stand up / settle" — any behavior rests -> l >= 0, seeding the
  # past-bar rest value. The easy fraction follows the assist knob down and
  # the hard fraction spawns under the beam, extending the win backward
  # (landing->launch trick). Without this a brake-competent policy parks at
  # the bar forever: the l gradient points through the beam but standing
  # tall strikes it.
  easy = u(0.0, 1.0) < (0.15 + 0.85 * assist)
  x_mid_hard = _BAR_X + BAR_DEPTH * u(0.0, 1.0)
  x_mid_easy = _BAR_X + BAR_DEPTH + u(0.4, 0.9)
  x = torch.where(midcrawl, torch.where(easy, x_mid_easy, x_mid_hard), x)
  z_mid = torch.minimum(u(0.15, 0.19), clearance - 0.075) - 0.32  # rel default z
  z_mid_easy = u(0.15, 0.19) - 0.32
  z = torch.where(midcrawl, torch.where(easy, z_mid_easy, z_mid), z)
  vx = torch.where(midcrawl, u(0.6, 1.6), vx)

  # crouch mask + depth: mid-crawl always; pre-crouched must-crawl too
  crouch = midcrawl | pre_crouched
  env._crouch_mask[env_ids] = crouch
  env._crouch_alpha[env_ids] = ((0.30 - clearance) / 0.08).clamp(0.0, 1.0)

  pose = torch.stack(
    [x, u(-0.06, 0.06), z,
     u(-0.05, 0.05), u(-0.05, 0.05), u(-0.06, 0.06)], dim=1)
  velocities = root[:, 7:13] + torch.stack(
    [vx, u(-0.10, 0.10), vz, u(-0.10, 0.10), u(-0.10, 0.10), u(-0.10, 0.10)],
    dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))

  # handover replay: real walker states placed on the approach
  if hdata is not None and bool(handover.any()):
    h_idx = handover.nonzero().flatten()
    lvl = env._handover_level[env_ids][h_idx].float()
    n_rows = len(hdata["z"])
    lo_f = 0.15 * lvl
    hi_f = (lo_f + 0.30).clamp(max=1.0)
    frac = torch.rand(len(h_idx), device=device)
    rows = ((lo_f + frac * (hi_f - lo_f)) * (n_rows - 1)).long()
    d_h = sample_uniform(0.25, 1.75, (len(h_idx),), device)
    positions[h_idx, 0] = (env.scene.env_origins[env_ids][h_idx, 0]
                           + _BAR_X - _NOSE - d_h)
    positions[h_idx, 1] = (env.scene.env_origins[env_ids][h_idx, 1]
                           + sample_uniform(-0.06, 0.06, (len(h_idx),), device))
    positions[h_idx, 2] = hdata["z"][rows]
    orientations[h_idx] = hdata["quat"][rows]
    velocities[h_idx, 0:3] = hdata["lin_vel_w"][rows]
    velocities[h_idx, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids[h_idx]] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids[h_idx]] = hdata["joint_vel"][rows]

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def apply_crouch_joints(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Write crouch joint poses for crouch-mask envs (runs AFTER
  reset_robot_joints so the default-pose randomization doesn't clobber it)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0 or not hasattr(env, "_crouch_mask"):
    return
  mask = env._crouch_mask[env_ids]
  if not bool(mask.any()):
    return
  asset = env.scene[asset_cfg.name]
  ids = env_ids[mask.nonzero().flatten()]
  m = int(len(ids))
  device = env.device
  alpha = env._crouch_alpha[ids].unsqueeze(-1)

  jpos = asset.data.default_joint_pos[ids].clone()
  thigh = _CROUCH_SHALLOW[0] + (_CROUCH_DEEP[0] - _CROUCH_SHALLOW[0]) * alpha
  calf = _CROUCH_SHALLOW[1] + (_CROUCH_DEEP[1] - _CROUCH_SHALLOW[1]) * alpha
  noise = sample_uniform(-0.05, 0.05, (m, 12), device)
  # Go2 joint layout: per-leg (hip, thigh, calf) x 4 legs.
  for leg in range(4):
    jpos[:, 3 * leg + 1] = thigh.squeeze(-1)
    jpos[:, 3 * leg + 2] = calf.squeeze(-1)
  jpos += noise
  jvel = sample_uniform(-0.5, 0.5, (m, 12), device)
  asset.write_joint_state_to_sim(jpos, jvel, env_ids=ids)


def apply_handover_joints_crawl(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0 or not hasattr(env, "_handover_mask"):
    return
  mask = env._handover_mask[env_ids]
  if not bool(mask.any()):
    return
  asset = env.scene[asset_cfg.name]
  ids = env_ids[mask.nonzero().flatten()]
  asset.write_joint_state_to_sim(
    env._handover_jpos[ids], env._handover_jvel[ids], env_ids=ids)


# --- curricula ----------------------------------------------------------------
# Curricula run at reset BEFORE the reset events, so robot state is still the
# episode's final state (same trick terrain_levels_parkour uses).

def _crossed_bar(env, env_ids) -> torch.Tensor:
  x_rel = (env.scene["robot"].data.root_link_pos_w[env_ids, 0]
           - env.scene.env_origins[env_ids, 0])
  return x_rel > (_BAR_X + BAR_DEPTH)


def crawl_assist_levels(env, env_ids) -> torch.Tensor:
  """Per-env reverse curriculum on the crawl: finish a crawl-stratum episode
  (must-crawl OR mid-crawl) PAST the bar -> promote (less assist: fewer easy
  cleared spawns, upright spawns further out); fall -> demote. Surviving by
  braking must NOT promote — that withdrew the pre-crouched rare-win foothold
  while the policy only knew how to stop."""
  _ensure_crawl_buffers(env)
  was = env._was_crawlwin[env_ids]
  t_o = env.termination_manager.time_outs[env_ids]
  crossed = _crossed_bar(env, env_ids)
  lvl = env._crawl_level[env_ids]
  lvl = torch.where(was & t_o & crossed, lvl + 1, lvl)
  lvl = torch.where(was & ~t_o, lvl - 1, lvl)
  env._crawl_level[env_ids] = lvl.clamp(0, _CRAWL_LEVELS)
  return env._crawl_level.float().mean()


def handover_levels_crawl(env, env_ids) -> torch.Tensor:
  _ensure_crawl_buffers(env)
  was = env._handover_mask[env_ids]
  t_o = env.termination_manager.time_outs[env_ids]
  crossed = _crossed_bar(env, env_ids)
  lvl = env._handover_level[env_ids]
  lvl = torch.where(was & t_o & crossed, lvl + 1, lvl)
  lvl = torch.where(was & ~t_o, lvl - 1, lvl)
  env._handover_level[env_ids] = lvl.clamp(0, 5)
  return env._handover_level.float().mean()


def crawl_filter_levels(env, env_ids) -> torch.Tensor | None:
  """Terrain gated on the binding skill: promote only when a must-crawl env
  actually finished PAST the bar. On impossible rows _was_mustcrawl never
  fires -> those rows only demote on falls; stopping there is trained without
  misreading it as crawl mastery."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  _ensure_crawl_buffers(env)
  t_o = env.termination_manager.time_outs[env_ids]
  was = env._was_mustcrawl[env_ids]
  crossed = _crossed_bar(env, env_ids)
  terrain.update_env_origins(env_ids, was & t_o & crossed, ~t_o)
  return terrain.terrain_levels.float().mean()


def pinned_levels_crawl(env, env_ids) -> torch.Tensor:
  """Adversarial phases: pin per-env curricula at mastered levels."""
  _ensure_crawl_buffers(env)
  env._crawl_level[env_ids] = 4
  env._handover_level[env_ids] = 3
  return env._crawl_level.float().mean()


# --- env cfg builders -----------------------------------------------------------

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
    term = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0}
    )
    cfg.observations["proprioception"].terms[name] = term
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0}
    )
  cfg.observations["critic"].terms["bar_info"] = ObservationTermCfg(
    func=mdp.bar_info, params={}
  )


def unitree_go2_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from src.tasks.go2_safety_filter.gap.env_cfg import (
    unitree_go2_gap_reach_avoid_env_cfg,
  )

  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CRAWL_FILTER_TERRAINS_CFG)
  cfg.episode_length_s = 8.0

  _add_bar_perception(cfg)

  cfg.events["reset_base"] = EventTermCfg(
    func=reset_takeover_crawl, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  # Ordering: crouch/handover joints AFTER reset_robot_joints; window last.
  cfg.events["crouch_joints"] = EventTermCfg(
    func=apply_crouch_joints, mode="reset", params={})
  cfg.events["handover_joints"] = EventTermCfg(
    func=apply_handover_joints_crawl, mode="reset", params={})
  cfg.events["rest_obstacle_window"] = EventTermCfg(
    func=set_rest_obstacle_window, mode="reset", params={})
  cfg.events.pop("push_robot", None)
  # The parkour base adds a per-reset terrain re-roll in play mode; it breaks
  # row pinning (the clearance rows ARE the benchmark axis) — drop it.
  cfg.events.pop("randomize_terrain", None)

  # Phase-offset-invariant gait clock (same rationale as crossing_chain).
  import copy as _copy
  for gname in ("proprioception", "critic"):
    term = _copy.deepcopy(cfg.observations[gname].terms["phase"])
    period = float(term.params.get("period", 0.5))
    term.func = phase_random_offset
    term.params = {"period": period}
    cfg.observations[gname].terms["phase"] = term

  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=crawl_filter_levels),
    "crawl_assist": CurriculumTermCfg(func=crawl_assist_levels),
    "handover_level": CurriculumTermCfg(func=handover_levels_crawl),
  }
  return cfg


def unitree_go2_crawl_isaacs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.events["reset_base"].params["stop_margin"] = 0.3
  cfg.curriculum = {"pinned_levels": CurriculumTermCfg(func=pinned_levels_crawl)}
  return cfg
