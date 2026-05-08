"""Hard-switch safety shield.

At each step, queries the safety critic for V_safe(s). If V_safe is above
``threshold`` the nominal walking action is used; otherwise the safety action
takes over. 

The caller provides both actor networks and the safety critic; all three
consume the same ``TensorDict`` observation (the shared proprioception group
is what lets this work without any obs adapter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDict


@dataclass
class ShieldStats:
  """Per-step shield telemetry (mean over num_envs)."""
  v_safe_mean: float = 0.0
  v_safe_min: float = 0.0
  safe_rate: float = 0.0       # fraction where V_safe > threshold
  override_rate: float = 0.0   # fraction where safety took over

  @staticmethod
  def from_step(v_safe: torch.Tensor, unsafe_mask: torch.Tensor) -> "ShieldStats":
    return ShieldStats(
      v_safe_mean=float(v_safe.mean().item()),
      v_safe_min=float(v_safe.min().item()),
      safe_rate=float((~unsafe_mask).float().mean().item()),
      override_rate=float(unsafe_mask.float().mean().item()),
    )


class ShieldedPolicy:
  """Runtime policy that hard-switches between walking and safety actions.

  Parameters
  ----------
  walk_actor:
      Callable returning nominal action for a given obs TensorDict. In
      practice this is ``runner_walk.get_inference_policy(device)``.
  safe_actor:
      Callable returning safety action for a given obs TensorDict.
  safe_critic:
      Callable returning V_safe(s) for a given obs TensorDict. Expected to
      be ``runner_safe.alg.critic`` after eval_mode().
  threshold:
      Scalar τ. Environments with ``V_safe <= τ`` are driven by the safety
      actor; others by the walking actor.
  """

  def __init__(
    self,
    walk_actor: Callable[[TensorDict], torch.Tensor],
    safe_actor: Callable[[TensorDict], torch.Tensor],
    safe_critic: Callable[[TensorDict], torch.Tensor],
    threshold: float = 0.05,
    log_fn: Callable[[ShieldStats], None] | None = None,
  ) -> None:
    self._walk = walk_actor
    self._safe = safe_actor
    self._critic = safe_critic
    self._threshold = float(threshold)
    self._log_fn = log_fn
    self.last_stats = ShieldStats()

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    v_safe = self._critic(obs).squeeze(-1)                    # (num_envs,)
    unsafe = v_safe <= self._threshold                        # bool mask

    a_walk = self._walk(obs)
    a_safe = self._safe(obs)

    action = torch.where(unsafe.unsqueeze(-1), a_safe, a_walk)

    self.last_stats = ShieldStats.from_step(v_safe, unsafe)
    if self._log_fn is not None:
      self._log_fn(self.last_stats)
    return action
