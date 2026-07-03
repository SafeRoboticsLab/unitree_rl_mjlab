"""ISAACS forced-crossing scenario: trapped-island Go2 gap reach-avoid.

The robot spawns on a short island (back pit behind, gap ahead, long safe far
platform beyond). Under the ISAACS adversary the small island has no robustly
stable stance, so the only V>0 option is to cross the (curriculum-widening) gap
forward and reach the far platform — the "fail unless you cross, while pushed by
a non-negatable adversary" scenario, learned end-to-end (no warm-start, no
reward shaping).

Margins:
  g(x) = min(terrain-relative height, tilt, non-foot contact)  (as parkour)
  l(x) = min(foothold, x_rel/L_NORM)  -- the reach target is the FAR platform
         ONLY (grounded AND past the island front edge). On the island (x_rel<0)
         or mid-gap (no foothold) l<0, so freezing is never a target.
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from src.isaacs_go2.go2_parkour_isaacs import (
  CENTRAL_RADIUS,
  CONTACT_FORCE_THRESHOLD,
  CTRL_DIM,
  DSTB_DIM,
  FOOTPRINT_RADIUS,
  HEIGHT_NORM,
  MIN_CLEARANCE,
  NONFOOT,
  OBSTACLE_MARGIN,
  SCAN,
  SIN_TILT_LIMIT,
  SUPPORT_NORM,
  SUPPORT_THRESHOLD,
  TERMINAL_MARGIN,
  _ForceMixin,
)
from src.isaacs_go2.island_terrain import ISLAND_CROSSING_TERRAINS_CFG
from src.tasks.go2_safety_filter.gap.env_cfg import unitree_go2_gap_reach_avoid_env_cfg

L_NORM = 0.5  # x_rel normalization for the far-platform reach margin

# --- Fixed-mix reset across the crossing manifold (offsets vs env_origin =
# island front edge; rows = 5 equally-sampled categories) ---
# cols: [x, y, z, roll, pitch, yaw]
_MIX_POSE_LOW = torch.tensor([
  [-0.55, -0.20, 0.00, -0.10, -0.10, -0.15],  # 0 standing-island
  [-0.15, -0.15, 0.00, -0.15, -0.10, -0.15],  # 1 committed-launch
  [0.10, -0.15, 0.15, -0.20, -0.25, -0.20],   # 2 mid-arc
  [0.50, -0.15, 0.05, -0.20, -0.20, -0.20],   # 3 near-landing
  [0.90, -0.20, 0.00, -0.10, -0.10, -0.15],   # 4 far-landed
  [0.05, -0.10, 0.25, -0.10, -0.15, -0.15],   # 5 midair-land (apex over gap, will clear)
])
_MIX_POSE_HIGH = torch.tensor([
  [-0.20, 0.20, 0.00, 0.10, 0.10, 0.15],
  [0.05, 0.15, 0.10, 0.15, 0.20, 0.15],
  [0.50, 0.15, 0.40, 0.20, 0.25, 0.20],
  [0.90, 0.15, 0.25, 0.20, 0.20, 0.20],
  [1.60, 0.20, 0.00, 0.10, 0.10, 0.15],
  [0.25, 0.10, 0.45, 0.10, 0.15, 0.15],
])
# cols: [vx, vy, vz, wx, wy, wz]
_MIX_VEL_LOW = torch.tensor([
  [-0.10, -0.10, -0.10, -0.10, -0.10, -0.10],  # 0 ~still
  [1.50, -0.20, 0.00, -0.30, -0.30, -0.30],    # 1 launch (forward+up)
  [1.50, -0.20, -1.00, -0.30, -0.30, -0.30],   # 2 mid-arc
  [0.50, -0.20, -1.50, -0.30, -0.30, -0.30],   # 3 near-landing (descending)
  [-0.10, -0.10, -0.10, -0.10, -0.10, -0.10],  # 4 ~still
  [2.50, -0.10, -0.50, -0.20, -0.20, -0.20],   # 5 midair-land (strong fwd, clears gap)
])
_MIX_VEL_HIGH = torch.tensor([
  [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
  [3.00, 0.20, 0.40, 0.30, 0.30, 0.30],
  [3.00, 0.20, 0.30, 0.30, 0.30, 0.30],
  [2.00, 0.20, -0.20, 0.30, 0.30, 0.30],
  [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
  [3.50, 0.10, 0.30, 0.20, 0.20, 0.20],
])


def reset_island_crossing_mix(env, env_ids, cats=None, asset_cfg=SceneEntityCfg("robot")):
  """Reset each env to one of 5 states across the crossing manifold (0 standing-
  island, 1 committed-launch, 2 mid-arc, 3 near-landing, 4 far-landed), so the
  reach-avoid value sees success (l>=0) states and their predecessors.

  ``cats=None`` samples all 5 equally (training). ``cats=[0]`` restricts to the
  standing-island start (eval: success then means a genuine cross from scratch).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  if cats is None:
    cat = torch.randint(0, 5, (n,), device=device)
  else:
    choices = torch.tensor(list(cats), device=device)
    cat = choices[torch.randint(0, len(choices), (n,), device=device)]
  pl, ph = _MIX_POSE_LOW.to(device)[cat], _MIX_POSE_HIGH.to(device)[cat]
  vl, vh = _MIX_VEL_LOW.to(device)[cat], _MIX_VEL_HIGH.to(device)[cat]
  pose = sample_uniform(pl, ph, (n, 6), device)
  vel = sample_uniform(vl, vh, (n, 6), device)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def island_margins(env):
  """(g, l) batched. g = safety; l = reach the FAR platform only."""
  robot = env.scene["robot"]
  scan = env.scene[SCAN]
  hit = scan.data.hit_pos_w
  dist = scan.data.distances
  base = robot.data.root_link_pos_w
  base_z, base_x = base[:, 2], base[:, 0]
  planar = torch.norm(hit[..., :2] - base[:, None, :2], dim=-1)
  hit_z = hit[..., 2]
  in_fp = (dist >= 0) & (planar <= FOOTPRINT_RADIUS)
  neg = torch.full_like(hit_z, -1.0e9)
  pos = torch.full_like(hit_z, 1.0e9)

  # g: terrain-relative base height above the local platform.
  below = in_fp & (hit_z < base_z[:, None] - OBSTACLE_MARGIN)
  ground_ref = torch.where(below, hit_z, neg).max(dim=1).values
  lowest = torch.where(in_fp, hit_z, pos).min(dim=1).values
  lowest = torch.where(in_fp.any(dim=1), lowest, base_z)
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

  # l: reach the FAR platform = grounded (foothold) AND past the island edge.
  central = (dist >= 0) & (planar <= CENTRAL_RADIUS)
  ground_under = torch.where(central, hit_z, neg).max(dim=1).values
  ground_under = torch.where(~central.any(dim=1), base_z - 5.0, ground_under)
  foothold = (SUPPORT_THRESHOLD - (base_z - ground_under)) / SUPPORT_NORM
  x_rel = (base_x - env.scene.env_origins[:, 0]) / L_NORM
  l = torch.minimum(foothold, x_rel)
  return g.clamp(-3.0, 3.0), l.clamp(-3.0, 3.0)


def island_hook(env) -> torch.Tensor:
  g, l = island_margins(env)
  failed = env.termination_manager.terminated
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, TERMINAL_MARGIN)), g)
  env.extras["isaacs_g"] = g
  env.extras["isaacs_l"] = l
  return g


def _build_island_cfg(num_envs: int, reset_mode: str = "mix"):
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=False)
  cfg.scene.num_envs = int(num_envs)
  cfg.scene.terrain.terrain_generator = replace(ISLAND_CROSSING_TERRAINS_CFG)
  cfg.episode_length_s = 6.0
  # Training: fixed-mix informative reset across the crossing manifold (value
  # sees success + predecessor states). Eval: island-only start, so success
  # measures a genuine cross from scratch.
  cats = {"island": [0], "midair_land": [5], "mix": None}.get(reset_mode, None)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_island_crossing_mix, mode="reset", params={"cats": cats}
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  # Zero command: the crossing is forced by the adversary + reach-avoid value,
  # not by a velocity command.
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if cfg.events is not None:
    cfg.events.pop("push_robot", None)
  cfg.rewards["isaacs_safety_hook"] = RewardTermCfg(func=island_hook, weight=1.0, params={})
  return cfg


class Go2IslandCrossingVecEnv(VecEnv, _ForceMixin):
  """Parallel SB3 VecEnv: trapped-island forced-crossing Go2 + force adversary."""

  def __init__(self, num_envs=64, device="cuda:0", render_mode=None, *,
               ctrl_gain=3.0, force_max=50.0, adversary=True, reset_mode="mix"):
    self._device = device
    self.render_mode = render_mode
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    self.adversary = bool(adversary)
    self.mj = ManagerBasedRlEnv(
      cfg=_build_island_cfg(num_envs, reset_mode), device=device,
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
    act_space = spaces.Box(-1.0, 1.0, shape=(CTRL_DIM + DSTB_DIM,), dtype=np.float32)
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
    ctrl = torch.as_tensor(a[:, :CTRL_DIM] * self.ctrl_gain, dtype=torch.float32,
                           device=self._device).reshape(self.num_envs, CTRL_DIM)
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


class Go2IslandCrossingEnv(gym.Env, _ForceMixin):
  """Single-env gym.Env version (for eval/video)."""

  metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
  CTRL_DIM = CTRL_DIM
  DSTB_DIM = DSTB_DIM

  def __init__(self, device="cuda:0", render_mode=None, *,
               ctrl_gain=3.0, force_max=50.0, adversary=True, reset_mode="mix"):
    super().__init__()
    self.render_mode = render_mode
    self._device = device
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    self.adversary = bool(adversary)
    self.mj = ManagerBasedRlEnv(
      cfg=_build_island_cfg(1, reset_mode), device=device,
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
    g, l = island_margins(self.mj)
    return self._obs(obs_dict), {"g_x": float(g[0]), "l_x": float(l[0])}

  def step(self, action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if self.adversary:
      self._set_force(action[CTRL_DIM:])
    ctrl = torch.tensor(action[:CTRL_DIM] * self.ctrl_gain, dtype=torch.float32,
                        device=self._device).reshape(1, CTRL_DIM)
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
