"""Parallel SB3 ``VecEnv`` over mjlab's native GPU-batched Go2 env (ISAACS Tier 2/3).

Unlike :class:`Go2IsaacsEnv` (one ``gym.Env`` that SB3 wraps in a
``DummyVecEnv(1)``), this exposes mjlab's *native* batched ``ManagerBasedRlEnv``
(``num_envs=N`` stepping together on GPU) directly through SB3's ``VecEnv``
interface. mjlab steps N envs almost as cheaply as one, so this is where the
GPU parallelism is actually used.

Off-policy SAC saturates well before mjlab's thousands-of-envs ceiling (the
replay buffer goes correlated and the update/transition ratio drops), so the
intended N here is moderate (~32-64), with ``train_freq``/``gradient_steps``
tuned in the train script. The on-policy "league" route to true mjlab scale is
in ``doc/research_note_onpolicy_league_reach_avoid.md``.

Action / margins / terminal-state handling are identical to
:class:`Go2IsaacsEnv` (shared :func:`isaacs_safety_hook`): action =
``[ctrl(12), dstb(3)]``, reward = g(s'), ``info["l_x"]`` = l(s'). Because the
reach-avoid terminal target zeroes the next-state term (``not_done=0``), the
auto-reset masking of the *next obs* on true terminations is harmless; only
timeouts bootstrap, and those carry ``TimeLimit.truncated`` so SB3 treats them
correctly.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mjlab.envs import ManagerBasedRlEnv

from src.isaacs_go2.go2_isaacs_env import _build_cfg

CTRL_DIM = 12
DSTB_DIM = 3


class Go2IsaacsVecEnv(VecEnv):
  """mjlab Go2 flat (num_envs=N) as an SB3 VecEnv with a per-env force adversary."""

  def __init__(
    self,
    num_envs: int = 64,
    device: str = "cuda:0",
    render_mode: str | None = None,
    *,
    ctrl_gain: float = 3.0,
    force_max: float = 50.0,
    adversary: bool = True,
    task: str = "stabilize",
    cmd_vx: float = 1.0,
  ) -> None:
    self._device = device
    self.render_mode = render_mode
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)  # current force magnitude (mutated by curriculum)
    self.adversary = bool(adversary)

    self.mj = ManagerBasedRlEnv(
      cfg=_build_cfg(num_envs, task, cmd_vx), device=device,
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

    self._actions: np.ndarray | None = None

  # ----------------------------------------------------------------- helpers
  def _obs(self, obs_dict) -> np.ndarray:
    return obs_dict["actor"].detach().float().cpu().numpy().astype(np.float32)

  def _set_force(self, a_dstb: np.ndarray | None) -> None:
    if a_dstb is None:
      forces = self._zero_wrench
    else:
      a = torch.as_tensor(a_dstb, dtype=torch.float32, device=self._device).reshape(
        self.num_envs, 3
      )
      unit = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)  # direction only
      forces = (unit * self.force_max).reshape(self.num_envs, 1, 3)  # magnitude == force_max
    self._robot.write_external_wrench_to_sim(
      forces, self._zero_wrench, body_ids=self._base_body_ids, env_ids=self._all_ids
    )

  # -------------------------------------------------------------- VecEnv api
  def reset(self):
    self._set_force(None)
    obs_dict, _ = self.mj.reset()
    return self._obs(obs_dict)

  def step_async(self, actions: np.ndarray) -> None:
    self._actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)

  def step_wait(self):
    a = self._actions
    a_ctrl, a_dstb = a[:, :CTRL_DIM], a[:, CTRL_DIM:]
    if self.adversary:
      self._set_force(a_dstb)
    ctrl = torch.as_tensor(
      a_ctrl * self.ctrl_gain, dtype=torch.float32, device=self._device
    ).reshape(self.num_envs, CTRL_DIM)

    obs_dict, _reward, terminated, truncated, extras = self.mj.step(ctrl)
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
        # mjlab already auto-reset env i; obs[i] is the reset obs. Reach-avoid
        # zeroes the next-state term on true terminals, and timeouts are flagged
        # so SB3 bootstraps them correctly.
        info["terminal_observation"] = obs[i]
        info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
      infos.append(info)
    return obs, g, dones, infos

  def close(self) -> None:
    try:
      self.mj.close()
    except Exception:
      pass

  def render(self):
    return self.mj.render()

  def get_images(self):
    frame = self.mj.render()
    return [frame for _ in range(self.num_envs)]

  # --- required VecEnv stubs (single batched env; per-sub-env access is n/a) ---
  def _indices(self, indices):
    if indices is None:
      return range(self.num_envs)
    if isinstance(indices, int):
      return [indices]
    return indices

  def get_attr(self, attr_name, indices=None):
    val = getattr(self, attr_name, None)
    return [val for _ in self._indices(indices)]

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
    return [None for _ in self._indices(indices)]

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._indices(indices)]
