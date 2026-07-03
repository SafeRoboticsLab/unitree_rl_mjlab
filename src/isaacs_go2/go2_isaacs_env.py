"""Single-env Gymnasium bridge: mjlab Go2 (flat) <-> SB3 ISAACS.

ISAACS Tier 2. Wraps ONE mjlab ``ManagerBasedRlEnv`` (Go2 flat, proprioception,
no heightfield) as a ``gymnasium.Env`` so the SB3 :class:`safety_sb3.IsaacsSAC`
two-player reach-avoid agent can train against a base force adversary.

**Faithful to ``safe_adaptation_dev`` (go2_mujoco_isaacs_v22_long)**:

* Action ``a = [a_ctrl(12), a_dstb(3)]`` in ``[-1, 1]``.
* Disturbance: the 3 dstb dims are a *direction*, **unit-normalized then scaled
  by ``force_max``** (so the magnitude is exactly ``force_max`` N, matching the
  reference, not the per-axis box that made our old adversary ~73% too strong).
  ``force_max`` is mutated by a training-time **curriculum** (5 -> 50 N).
* ``g(s)`` (safety) = ``corner_height`` = ``min_i(z_i) - 0.10`` over the 8 trunk
  bounding-box corners (a single smooth tip/crash signal), clamped negative on
  failure terminations.
* ``l(s)`` (reach = stabilize-to-stance) = ``min`` of: corner-height band
  (``min_c - 0.10``, ``0.40 - max_c``), low linear velocity (``0.2 - |v|`` per
  axis), low angular velocity (``0.174 - |w|`` per axis). Positive = recovered
  to a stable upright stand.
* The velocity command is zeroed (the reference is a stabilization task, not
  command tracking).

Terminal margins are computed by a reward term (:func:`isaacs_safety_hook`) that
runs *before* mjlab's internal auto-reset and is stashed in ``env.extras``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg

# Go2 trunk bounding-box half-extents (m): length, width, height.
TRUNK_HALF = (0.18, 0.045, 0.05)
CORNER_LOW = 0.10   # min trunk corner must stay >= this (g threshold)
CORNER_HIGH = 0.40  # max trunk corner must stay <= this (not airborne)
LIN_TOL = 0.20      # |linear velocity| <= this at the stance target (m/s)
ANG_TOL = 0.174     # |angular velocity| <= this at the stance target (rad/s)
FAIL_G = -1.0       # g clamped at/below this on any failure termination


def _corner_offsets(device, dtype):
  hl, hw, hh = TRUNK_HALF
  pts = [[sx * hl, sy * hw, sz * hh] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
  return torch.tensor(pts, device=device, dtype=dtype)  # (8, 3)


def corner_stance_margins(d):
  """Return (g, l) batched over envs from robot data ``d`` (reference setup)."""
  pos = d.root_link_pos_w  # (N, 3)
  quat = d.root_link_quat_w  # (N, 4) wxyz
  v_b = d.root_link_lin_vel_b  # (N, 3)
  w_b = d.root_link_ang_vel_b  # (N, 3)
  n = pos.shape[0]

  offs = _corner_offsets(pos.device, pos.dtype)  # (8, 3)
  q = quat[:, None, :].expand(n, 8, 4).reshape(-1, 4)
  o = offs[None].expand(n, 8, 3).reshape(-1, 3)
  corners_w = pos[:, None, :].expand(n, 8, 3).reshape(-1, 3) + quat_apply(q, o)
  cz = corners_w.reshape(n, 8, 3)[..., 2]  # (N, 8) world-z of each corner
  min_c = cz.amin(dim=1)
  max_c = cz.amax(dim=1)

  g = min_c - CORNER_LOW
  terms = torch.stack(
    [
      min_c - CORNER_LOW,
      CORNER_HIGH - max_c,
      LIN_TOL - v_b[:, 0].abs(),
      LIN_TOL - v_b[:, 1].abs(),
      LIN_TOL - v_b[:, 2].abs(),
      ANG_TOL - w_b[:, 0].abs(),
      ANG_TOL - w_b[:, 1].abs(),
      ANG_TOL - w_b[:, 2].abs(),
    ],
    dim=0,
  )
  l = terms.amin(dim=0)
  return g, l


def isaacs_safety_hook(env) -> torch.Tensor:
  """mjlab reward term used purely as a *pre-reset hook* (see module docstring).

  Stashes terminal-correct g/l in ``env.extras`` before mjlab's auto-reset. g is
  clamped to ``FAIL_G`` on failure terminations (``illegal_contact`` etc. that
  the geometric margin may not catch), keeping the failure set coincident with
  ``g < 0``.
  """
  g, l = corner_stance_margins(env.scene["robot"].data)
  failed = env.termination_manager.terminated
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, FAIL_G)), g)
  env.extras["isaacs_g"] = g
  env.extras["isaacs_l"] = l
  return g


# --- locomote (step-1 diagnostic) margin constants ---
Z_MIN = 0.18
Z_SCALE = 0.15
TILT_COS = 0.64   # cos(~50deg)
TILT_NORM = 1.0 - TILT_COS
LOCO_VEL_TOL = 0.5  # |v_xy - cmd_xy| <= this == "tracking the command" (reach)


def locomote_margins(env):
  """Forward-locomotion reach-avoid margins (step-1 diagnostic).

  g = min(base-height, tilt): stay upright and off the floor (robust during a
  dynamic gait, unlike the corner signal which can clip mid-stride).
  l = LOCO_VEL_TOL - ||v_xy - cmd_xy||: the reach target is *tracking the
  forward velocity command* (so V>0 requires the robot can safely move at the
  command). Crucially the reach is moving, not standing still.
  """
  d = env.scene["robot"].data
  base_z = d.root_link_pos_w[:, 2]
  up = -d.projected_gravity_b[:, 2]
  v_b = d.root_link_lin_vel_b[:, :2]
  cmd = env.command_manager.get_command("twist")[:, :2]
  height = (base_z - Z_MIN) / Z_SCALE
  tilt = (up - TILT_COS) / TILT_NORM
  g = torch.minimum(height, tilt)
  l = LOCO_VEL_TOL - torch.linalg.norm(v_b - cmd, dim=1)
  return g, l


def isaacs_locomote_hook(env) -> torch.Tensor:
  g, l = locomote_margins(env)
  failed = env.termination_manager.terminated
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, FAIL_G)), g)
  env.extras["isaacs_g"] = g
  env.extras["isaacs_l"] = l
  return g


def _build_cfg(num_envs: int, task: str = "stabilize", cmd_vx: float = 1.0):
  cfg = unitree_go2_flat_env_cfg(play=False)
  cfg.scene.num_envs = int(num_envs)
  if cfg.events is not None:
    cfg.events.pop("push_robot", None)  # learned adversary replaces random push
  twist = cfg.commands["twist"]
  if task == "locomote":
    twist.ranges.lin_vel_x = (cmd_vx, cmd_vx)  # constant forward drive
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(twist, "rel_standing_envs"):
      twist.rel_standing_envs = 0.0
    if hasattr(twist, "heading_command"):
      twist.heading_command = False
      if getattr(twist.ranges, "heading", None) is not None:
        twist.ranges.heading = None  # required when heading_command is disabled
    hook = isaacs_locomote_hook
  else:  # stabilize (reference setup): zero command, stance target
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    hook = isaacs_safety_hook
  cfg.rewards["isaacs_safety_hook"] = RewardTermCfg(func=hook, weight=1.0, params={})
  return cfg


def _margins_for(task):
  return locomote_margins if task == "locomote" else (lambda env: corner_stance_margins(env.scene["robot"].data))


class Go2IsaacsEnv(gym.Env):
  """mjlab Go2 flat env as a single-agent Gymnasium env with a force adversary."""

  metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

  CTRL_DIM = 12
  DSTB_DIM = 3

  def __init__(
    self,
    device: str = "cuda:0",
    render_mode: str | None = None,
    *,
    ctrl_gain: float = 3.0,
    force_max: float = 50.0,
    adversary: bool = True,
    task: str = "stabilize",
    cmd_vx: float = 1.0,
  ) -> None:
    super().__init__()
    self.render_mode = render_mode
    self._device = device
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)  # current force magnitude (mutated by curriculum)
    self.adversary = bool(adversary)
    self._margin_fn = _margins_for(task)

    self.mj = ManagerBasedRlEnv(
      cfg=_build_cfg(1, task, cmd_vx), device=device,
      render_mode="rgb_array" if render_mode else None,
    )
    self._robot = self.mj.scene["robot"]
    body_ids, _ = self._robot.find_bodies("base_link")
    self._base_body_ids = list(body_ids)
    self._env0 = torch.tensor([0], device=self._device)

    obs_dict, _ = self.mj.reset()
    obs0 = self._extract_obs(obs_dict)
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs0.shape, dtype=np.float32)
    self.action_space = gym.spaces.Box(
      -1.0, 1.0, shape=(self.CTRL_DIM + self.DSTB_DIM,), dtype=np.float32
    )

  # ----------------------------------------------------------------- helpers
  def _extract_obs(self, obs_dict) -> np.ndarray:
    return obs_dict["actor"].detach().float().cpu().numpy().reshape(-1).astype(np.float32)

  def _apply_disturbance(self, a_dstb: np.ndarray | None) -> None:
    if a_dstb is None:
      forces = torch.zeros((1, 1, 3), device=self._device)
    else:
      a = torch.as_tensor(a_dstb, dtype=torch.float32, device=self._device).reshape(1, 3)
      unit = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)  # direction only
      forces = (unit * self.force_max).reshape(1, 1, 3)       # magnitude == force_max
    self._robot.write_external_wrench_to_sim(
      forces, torch.zeros_like(forces), body_ids=self._base_body_ids, env_ids=self._env0
    )

  # -------------------------------------------------------------------- gym api
  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self._apply_disturbance(None)
    obs_dict, _ = self.mj.reset()
    obs = self._extract_obs(obs_dict)
    g, l = self._margin_fn(self.mj)
    info = {"g_x": float(g[0]), "l_x": float(l[0])}
    return obs, info

  def step(self, action: np.ndarray):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if self.adversary:
      self._apply_disturbance(action[self.CTRL_DIM :])
    ctrl = torch.tensor(
      action[: self.CTRL_DIM] * self.ctrl_gain, dtype=torch.float32, device=self._device
    ).reshape(1, self.CTRL_DIM)
    obs_dict, _reward, terminated, truncated, extras = self.mj.step(ctrl)

    obs = self._extract_obs(obs_dict)
    g = float(extras["isaacs_g"][0].item())  # pre-reset (terminal-correct)
    l = float(extras["isaacs_l"][0].item())
    return obs, g, bool(terminated[0]), bool(truncated[0]), {"g_x": g, "l_x": l}

  def render(self):
    return self.mj.render()

  def close(self):
    try:
      self.mj.close()
    except Exception:
      pass
