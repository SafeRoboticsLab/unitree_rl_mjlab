"""ISAACS Tier 3, Stage 1: SB3 ISAACS on the Go2 gap-crossing reach-avoid task.

Wraps the existing ``unitree_go2_gap_reach_avoid_env_cfg`` (GAP_EDGE terrain,
proprioception + ``height_scan`` MLP actor, depth dropped, midair-over-gaps
reset) as an SB3 ``VecEnv`` / ``gym.Env`` so the validated two-player
:class:`safety_sb3.IsaacsSAC` can train a gap-crossing safety controller against
a base-force adversary (curriculum 5 -> 50 N).

The reach-avoid margins are the parkour ones (ported verbatim from
``ParkourReachAvoidVecEnvWrapper`` + ``GapReachAvoidVecEnvWrapper``):

    g(x) = min( terrain-relative base-height , tilt , non-foot-contact )
    l(x) = foothold-support + 0.5*forward-progress   (gap reach: stop at an
           uncrossable gap is itself a target; landing restores l>=0)

``g``/``l`` are computed by a reward-term hook that runs *before* mjlab's
auto-reset (terminal-correct), mirroring the flat-ground Tier-2 design.

Stage 2 (deployable) will swap the proprio+height_scan MLP for a depth-CNN
multi-input policy; this stage validates the gap reach-avoid game first.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.go2_safety_filter.gap.env_cfg import unitree_go2_gap_reach_avoid_env_cfg

CTRL_DIM = 12
DSTB_DIM = 3

# --- margin constants (from ParkourReachAvoidVecEnvWrapper / GapReachAvoid...) ---
MIN_CLEARANCE = 0.08
HEIGHT_NORM = 0.25
SIN_TILT_LIMIT = math.sin(math.radians(70.0))
OBSTACLE_MARGIN = 0.12
FOOTPRINT_RADIUS = 0.5
CONTACT_FORCE_THRESHOLD = 10.0
CENTRAL_RADIUS = 0.20
SUPPORT_THRESHOLD = 0.45
SUPPORT_NORM = 0.30
PROGRESS_WEIGHT = 0.5
PATCH_LENGTH = 8.0
TERMINAL_MARGIN = -0.1  # g anchor on failure termination (matches the wrapper)
SCAN = "terrain_scan"
NONFOOT = "nonfoot_ground_touch"


def parkour_gap_margins(env):
  """Return (g, l) batched over envs (terrain-relative reach-avoid, gap reach)."""
  robot = env.scene["robot"]
  scan = env.scene[SCAN]
  hit = scan.data.hit_pos_w  # (B, N, 3)
  dist = scan.data.distances  # (B, N); <0 == miss

  base_z = robot.data.root_link_pos_w[:, 2]  # (B,)
  base_xy = robot.data.root_link_pos_w[:, None, :2]  # (B, 1, 2)
  planar = torch.norm(hit[..., :2] - base_xy, dim=-1)  # (B, N)
  hit_z = hit[..., 2]
  in_footprint = (dist >= 0) & (planar <= FOOTPRINT_RADIUS)

  # --- g: terrain-relative base height above the LOCAL platform ---
  below = in_footprint & (hit_z < base_z[:, None] - OBSTACLE_MARGIN)
  neg_inf = torch.full_like(hit_z, -1.0e9)
  pos_inf = torch.full_like(hit_z, 1.0e9)
  ground_ref = torch.where(below, hit_z, neg_inf).max(dim=1).values
  lowest = torch.where(in_footprint, hit_z, pos_inf).min(dim=1).values
  lowest = torch.where(in_footprint.any(dim=1), lowest, base_z)
  ground_ref = torch.where(below.any(dim=1), ground_ref, lowest)
  height = (base_z - ground_ref - MIN_CLEARANCE) / HEIGHT_NORM

  grav_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  orientation = (SIN_TILT_LIMIT - grav_xy) / SIN_TILT_LIMIT

  g_terms = [height, orientation]
  try:
    sensor = env.scene[NONFOOT]
    force = sensor.data.force_history if sensor.data.force_history is not None else sensor.data.force
    if force is not None:
      mag = torch.norm(force, dim=-1)
      while mag.dim() > 1:
        mag = mag.amax(dim=-1)
      g_terms.append((CONTACT_FORCE_THRESHOLD - mag) / CONTACT_FORCE_THRESHOLD)
  except (KeyError, AttributeError):
    pass
  g = torch.stack(g_terms, dim=-1).min(dim=-1).values

  # --- l: foothold support directly under the robot + forward progress ---
  central = (dist >= 0) & (planar <= CENTRAL_RADIUS)
  ground_under = torch.where(central, hit_z, neg_inf).max(dim=1).values
  ground_under = torch.where(~central.any(dim=1), base_z - 5.0, ground_under)
  support_distance = base_z - ground_under
  foothold = (SUPPORT_THRESHOLD - support_distance) / SUPPORT_NORM
  origin_x = env.scene.env_origins[:, 0]
  progress = ((robot.data.root_link_pos_w[:, 0] - origin_x) / PATCH_LENGTH).clamp(0.0, 1.5)
  l = foothold + PROGRESS_WEIGHT * progress
  # Bound the dynamic range for SAC critic regression. The /HEIGHT_NORM, /SUPPORT_NORM
  # scaling makes doomed over-gap flight states reach ~-26; clamping preserves the
  # sign (safe/reached membership) while keeping value targets well-scaled.
  return g.clamp(-3.0, 3.0), l.clamp(-3.0, 3.0)


def parkour_isaacs_hook(env, margin_fn=None) -> torch.Tensor:
  """Pre-reset reward-term hook: stash terminal-correct g/l in env.extras.

  ``margin_fn(env) -> (g, l)`` selects the task's reach-avoid margins
  (default: the gap foothold+progress margins). g is anchored to the terminal
  failure value on real terminations, matching the rsl_rl wrapper."""
  if margin_fn is None:
    margin_fn = parkour_gap_margins
  g, l = margin_fn(env)
  failed = env.termination_manager.terminated
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, TERMINAL_MARGIN)), g)
  env.extras["isaacs_g"] = g
  env.extras["isaacs_l"] = l
  return g


def _build_parkour_cfg(num_envs: int, cfg_builder=None, margin_fn=None):
  """Build an mjlab env cfg for the SB3 bridge: any go2_safety_filter cfg
  builder + the isaacs g/l reward hook (the env logic — spawn strata, reverse
  curricula, handover — is algorithm-agnostic and reused as-is)."""
  if cfg_builder is None:
    cfg_builder = unitree_go2_gap_reach_avoid_env_cfg
  cfg = cfg_builder(play=False)
  cfg.scene.num_envs = int(num_envs)
  if cfg.events is not None:
    cfg.events.pop("push_robot", None)  # learned force adversary replaces it
  cfg.rewards["isaacs_safety_hook"] = RewardTermCfg(
    func=parkour_isaacs_hook, weight=1.0, params={"margin_fn": margin_fn})
  return cfg


class _ForceMixin:
  """Shared unit-normalized base-force adversary (magnitude == force_max)."""

  def _force_tensor(self, a_dstb, n):
    a = torch.as_tensor(a_dstb, dtype=torch.float32, device=self._device).reshape(n, 3)
    unit = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (unit * self.force_max).reshape(n, 1, 3)


class Go2ParkourIsaacsVecEnv(VecEnv, _ForceMixin):
  """Parallel SB3 VecEnv over the mjlab Go2 gap reach-avoid env + force adversary."""

  def __init__(self, num_envs=64, device="cuda:0", render_mode=None, *,
               ctrl_gain=3.0, force_max=50.0, adversary=True,
               cfg_builder=None, margin_fn=None):
    self._device = device
    self.render_mode = render_mode
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    self.adversary = bool(adversary)

    self.mj = ManagerBasedRlEnv(
      cfg=_build_parkour_cfg(num_envs, cfg_builder=cfg_builder, margin_fn=margin_fn),
      device=device,
      render_mode="rgb_array" if render_mode else None,
    )
    self._robot = self.mj.scene["robot"]
    body_ids, _ = self._robot.find_bodies("base_link")
    self._base_body_ids = list(body_ids)
    self._all_ids = torch.arange(int(num_envs), device=device)
    self._zero_wrench = torch.zeros((int(num_envs), 1, 3), device=device)

    obs_dict, _ = self.mj.reset()
    obs0 = self._obs(obs_dict)
    obs_space = spaces.Box(-np.inf, np.inf, shape=(obs0.shape[1],), dtype=np.float32)
    # Single-agent (adversary off) exposes a ctrl-only action space; the
    # two-player ISAACS action appends the DSTB force dims. step_wait reads
    # a[:, :CTRL_DIM] for ctrl and only touches a[:, CTRL_DIM:] when adversary.
    act_dim = CTRL_DIM + (DSTB_DIM if self.adversary else 0)
    act_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
    super().__init__(int(num_envs), obs_space, act_space)
    self._actions = None

  def _obs(self, obs_dict) -> np.ndarray:
    return obs_dict["proprioception"].detach().float().cpu().numpy().astype(np.float32)

  def _set_force(self, a_dstb):
    forces = self._zero_wrench if a_dstb is None else self._force_tensor(a_dstb, self.num_envs)
    self._robot.write_external_wrench_to_sim(
      forces, self._zero_wrench, body_ids=self._base_body_ids, env_ids=self._all_ids
    )

  def reset(self):
    self._set_force(None)
    obs_dict, _ = self.mj.reset()
    return self._obs(obs_dict)

  def step_async(self, actions):
    self._actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)

  def step_wait(self):
    a = self._actions
    if self.adversary:
      self._set_force(a[:, CTRL_DIM:])
    ctrl = torch.as_tensor(
      a[:, :CTRL_DIM] * self.ctrl_gain, dtype=torch.float32, device=self._device
    ).reshape(self.num_envs, CTRL_DIM)
    obs_dict, _r, terminated, truncated, extras = self.mj.step(ctrl)
    obs = self._obs(obs_dict)
    g = extras["isaacs_g"].detach().float().cpu().numpy().astype(np.float32)
    l = extras["isaacs_l"].detach().float().cpu().numpy().astype(np.float32)
    term = terminated.detach().cpu().numpy()
    trunc = truncated.detach().cpu().numpy()
    dones = np.logical_or(term, trunc)
    infos = []
    for i in range(self.num_envs):
      info = {"l_x": float(l[i])}
      if dones[i]:
        info["terminal_observation"] = obs[i]
        info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
      infos.append(info)
    return obs, g, dones, infos

  def close(self):
    try:
      self.mj.close()
    except Exception:
      pass

  def render(self):
    return self.mj.render()

  def get_images(self):
    return [self.mj.render() for _ in range(self.num_envs)]

  def _indices(self, indices):
    if indices is None:
      return range(self.num_envs)
    return [indices] if isinstance(indices, int) else indices

  def get_attr(self, attr_name, indices=None):
    return [getattr(self, attr_name, None) for _ in self._indices(indices)]

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *args, indices=None, **kwargs):
    return [None for _ in self._indices(indices)]

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._indices(indices)]


class Go2ParkourIsaacsEnv(gym.Env, _ForceMixin):
  """Single-env gym.Env version (for eval/video)."""

  metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
  CTRL_DIM = CTRL_DIM
  DSTB_DIM = DSTB_DIM

  def __init__(self, device="cuda:0", render_mode=None, *,
               ctrl_gain=3.0, force_max=50.0, adversary=True):
    super().__init__()
    self.render_mode = render_mode
    self._device = device
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    self.adversary = bool(adversary)
    self.mj = ManagerBasedRlEnv(
      cfg=_build_parkour_cfg(1), device=device,
      render_mode="rgb_array" if render_mode else None,
    )
    self._robot = self.mj.scene["robot"]
    body_ids, _ = self._robot.find_bodies("base_link")
    self._base_body_ids = list(body_ids)
    self._env0 = torch.tensor([0], device=device)
    obs_dict, _ = self.mj.reset()
    obs0 = self._obs(obs_dict)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs0.shape, dtype=np.float32)
    self.action_space = spaces.Box(-1.0, 1.0, shape=(CTRL_DIM + DSTB_DIM,), dtype=np.float32)

  def _obs(self, obs_dict) -> np.ndarray:
    return obs_dict["proprioception"].detach().float().cpu().numpy().reshape(-1).astype(np.float32)

  def _set_force(self, a_dstb):
    forces = torch.zeros((1, 1, 3), device=self._device) if a_dstb is None else self._force_tensor(a_dstb, 1)
    self._robot.write_external_wrench_to_sim(
      forces, torch.zeros_like(forces), body_ids=self._base_body_ids, env_ids=self._env0
    )

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self._set_force(None)
    obs_dict, _ = self.mj.reset()
    g, l = parkour_gap_margins(self._robot_env())
    return self._obs(obs_dict), {"g_x": float(g[0]), "l_x": float(l[0])}

  def _robot_env(self):
    return self.mj

  def step(self, action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if self.adversary:
      self._set_force(action[CTRL_DIM:])
    ctrl = torch.tensor(
      action[:CTRL_DIM] * self.ctrl_gain, dtype=torch.float32, device=self._device
    ).reshape(1, CTRL_DIM)
    obs_dict, _r, terminated, truncated, extras = self.mj.step(ctrl)
    g = float(extras["isaacs_g"][0].item())
    l = float(extras["isaacs_l"][0].item())
    return self._obs(obs_dict), g, bool(terminated[0]), bool(truncated[0]), {"g_x": g, "l_x": l}

  def render(self):
    return self.mj.render()

  def close(self):
    try:
      self.mj.close()
    except Exception:
      pass


# ---------------------------------------------------------------------------
# Separate SB3 VecEnv factories for the jumping pipeline (landing -> crossing ->
# chain), mirroring the rsl_rl task structure. The mjlab env cfgs carry the
# spawn strata + reverse curricula + handover; the bridge only swaps the cfg
# and (for the rest objective) the l margin. Avoid-only stages (landing,
# crossing) train with safety_sb3 SafetyPPO and ignore l; the chain stage uses
# ReachAvoidPPO with rest_margins.
# ---------------------------------------------------------------------------

def rest_margins(env):
  """Reach-avoid margins for the REST objective (crossing-chain): same g as the
  gap task; l = come to a safe stop + mild forward-progress cross bias
  (mirrors ParkourReachAvoidVecEnvWrapper rest mode: v_rest 0.3, norm 0.5,
  cross_bias 0.3, scale 3.0)."""
  g, _ = parkour_gap_margins(env)
  robot = env.scene["robot"]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  l_rest = (0.3 - speed) / 0.5
  origin_x = env.scene.env_origins[:, 0]
  prog = ((robot.data.root_link_pos_w[:, 0] - origin_x) / 3.0).clamp(0.0, 1.0)
  l = l_rest + 0.3 * prog
  return g.clamp(-3.0, 3.0), l.clamp(-3.0, 3.0)


def make_landing_vecenv(num_envs=1024, device="cuda:0", **kw):
  """Avoid-only landing (mid-air-over-gap spawn -> soft land). SafetyPPO."""
  from src.tasks.go2_safety_filter.landing.env_cfg import (
    unitree_go2_landing_env_cfg,
  )
  return Go2ParkourIsaacsVecEnv(
    num_envs=num_envs, device=device, adversary=False,
    cfg_builder=unitree_go2_landing_env_cfg, **kw)


def make_crossing_vecenv(num_envs=1024, device="cuda:0", **kw):
  """Avoid-only reverse-curriculum crossing (launch->land). SafetyPPO."""
  from src.tasks.go2_safety_filter.crossing.env_cfg import (
    unitree_go2_crossing_env_cfg,
  )
  return Go2ParkourIsaacsVecEnv(
    num_envs=num_envs, device=device, adversary=False,
    cfg_builder=unitree_go2_crossing_env_cfg, **kw)


def make_chain_vecenv(num_envs=1024, device="cuda:0", adversary=False, **kw):
  """Rest-objective crossing-chain (arrival momentum -> safe rest). ReachAvoid/
  IsaacsPPO (adversary=True for the two-player game)."""
  from src.tasks.go2_safety_filter.crossing_chain.env_cfg import (
    unitree_go2_crossing_chain_env_cfg,
  )
  return Go2ParkourIsaacsVecEnv(
    num_envs=num_envs, device=device, adversary=adversary,
    cfg_builder=unitree_go2_crossing_chain_env_cfg, margin_fn=rest_margins, **kw)
