"""Adversarial reach-avoid wrapper: per-step disturbance force channel.

Extends the rest-mode reach-avoid wrapper with a disturbance input: a 3-D
force *direction* on ``base_link``, unit-normalized and scaled to
``force_max`` Newtons (the game's disturbance bound; per-env ``force_scale``
in [0,1] is a training aid).  The runner sets the disturbance action before
each step; the wrapper converts and writes the wrench for ALL envs every step
(``xfrc_applied`` persists until rewritten — full-batch writes are the only
safe ownership discipline).

This wrapper must be the wrench channel's SOLE owner: it refuses to wrap an
env whose cfg carries the increment-0 ``set_random_push`` reset event.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.parkour.rl.reach_avoid_vecenv_wrapper import (
  ParkourReachAvoidVecEnvWrapper,
)


class AdversarialReachAvoidVecEnvWrapper(ParkourReachAvoidVecEnvWrapper):
  def __init__(self, env, *, force_max: float = 50.0, **kwargs) -> None:
    super().__init__(env, **kwargs)
    events = getattr(self.unwrapped.cfg, "events", {}) or {}
    assert "set_random_push" not in events, (
      "Adversarial wrapper owns the wrench channel exclusively; register the "
      "PLAIN crossing-chain env cfg (no set_random_push event)."
    )
    self._force_max = float(force_max)
    robot = self.unwrapped.scene["robot"]
    self._adv_body_ids = robot.find_bodies("base_link")[0]
    n = self.unwrapped.num_envs
    dev = self.unwrapped.device
    self._dstb_action = torch.zeros(n, 3, device=dev)
    self._force_scale = torch.ones(n, device=dev)

  # --- runner-facing knobs --------------------------------------------------

  def set_dstb_action(self, action: torch.Tensor | None) -> None:
    """Raw dstb action (N,3) for the NEXT step; None -> zero disturbance."""
    if action is None:
      self._dstb_action.zero_()
    else:
      self._dstb_action.copy_(action.detach())

  def set_force_scale(self, scale: torch.Tensor | float) -> None:
    """Per-env magnitude scale in [0,1] (curriculum/training aid)."""
    if isinstance(scale, torch.Tensor):
      self._force_scale.copy_(scale.clamp(0.0, 1.0))
    else:
      self._force_scale.fill_(min(max(float(scale), 0.0), 1.0))

  @property
  def force_max(self) -> float:
    return self._force_max

  # --- step -----------------------------------------------------------------

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    robot = self.unwrapped.scene["robot"]
    direction = self._dstb_action / self._dstb_action.norm(
      dim=1, keepdim=True
    ).clamp_min(1e-6)
    magnitude = (self._force_scale * self._force_max).unsqueeze(-1)
    forces = (direction * magnitude).unsqueeze(1)  # (N, 1 body, 3)
    torques = torch.zeros_like(forces)
    robot.write_external_wrench_to_sim(
      forces, torques, body_ids=self._adv_body_ids
    )

    obs, g, dones, extras = super().step(actions)
    log = extras.get("log", {})
    log["Dstb/force_mag_mean"] = magnitude.mean()
    log["Dstb/force_z_frac"] = direction[:, 2].abs().mean()
    extras["log"] = log
    return obs, g, dones, extras
