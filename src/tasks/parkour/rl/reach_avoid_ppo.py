"""Reach-avoid PPO with the Fisac-Chen Bellman backup.

The stock :class:`SafetyPPO` is *avoid-only*::

    V(x) = min(g(x), V(x'))

It can answer "what is the worst safety margin we will see?" but has no notion
of a target — every state with positive ``g`` looks equally good, so the optimal
policy is the most conservative one (e.g. never attempt a gap).  For
goal-conditioned parkour we need the **reach-avoid** Bellman backup::

    V(x) = min( g(x), max( l(x), γ·V(x') ) )

where ``g(x) > 0`` defines the physical safe set and ``l(x) > 0`` the target set.
``V(x) > 0`` iff there exists a policy that drives the state into ``{l > 0}``
(e.g. across the gap / tracking the velocity command) while never leaving
``{g > 0}`` (not falling, colliding, or tipping over).

References:
  J. F. Fisac et al., "Reach-Avoid Problems with Time-Varying Dynamics, Targets
  and Constraints", HSCC 2015.
  K.-C. Hsu, V. R. Royo, C. J. Tomlin, J. F. Fisac, "Safety and Liveness
  Guarantees through Reach-Avoid Reinforcement Learning", RSS 2021.

This subclass extends :class:`SafetyPPO`:

* ``process_env_step`` reads ``extras["target_margin"]`` (produced by the paired
  reach-avoid vec-env wrapper) and stores it in a parallel rollout buffer of
  shape ``(num_transitions_per_env, num_envs)``.
* ``compute_returns`` uses the reach-avoid backup above.

Everything else (PPO surrogate, value loss, KL schedule, clipping, entropy) is
inherited from the base PPO.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.parkour.rl.safety_ppo import SafetyPPO


class ReachAvoidPPO(SafetyPPO):
  """PPO with reach-avoid Bellman backup."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # ``storage`` is constructed inside ``PPO.__init__`` with its
    # ``num_transitions_per_env`` already set, so we can allocate the
    # parallel target-margin buffer here.
    st = self.storage
    self._target_margins: torch.Tensor = torch.zeros(
      st.num_transitions_per_env, st.num_envs, device=st.device
    )

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record (g, l) for the current step.

    ``rewards`` is ``g(x)`` (set by the reach-avoid wrapper).  The target
    margin ``l(x)`` is read from ``extras["target_margin"]``; if missing
    (e.g. an unwrapped env), we fall back to a large positive value so the
    reach-avoid backup degenerates to the avoid-only behaviour of the parent.
    """
    step = self.storage.step  # read before super() bumps it
    super().process_env_step(obs, rewards, dones, extras)

    if "target_margin" in extras:
      ell = extras["target_margin"].to(self.device)
    else:
      ell = torch.full_like(rewards, float("inf"))
    self._target_margins[step] = ell.detach()

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute returns using the reach-avoid Bellman backup.

    For each transition::

        g_t   = stored reward (safety margin)
        l_t   = stored target margin
        V_nx  = critic(s_{t+1})         (or last_values for t = T-1)
        V*_t  = min( g_t, max( l_t, γ·(1-d_t)·V_nx ) )       # non-terminal
        V*_t  = g_t                                            # terminal

    Then GAE-lambda smoothing is applied over the ``δ_t = V*_t - V_t``
    sequence, identical to standard PPO except the per-step target is the
    reach-avoid backup instead of ``r + γ V_nx``.
    """
    st = self.storage
    last_values = self.critic(obs).detach()

    advantage: torch.Tensor | float = 0
    for step in reversed(range(st.num_transitions_per_env)):
      if step == st.num_transitions_per_env - 1:
        next_values = last_values
      else:
        next_values = st.values[step + 1]

      next_is_not_terminal = 1.0 - st.dones[step].float()
      g_t = st.rewards[step]
      ell_t = self._target_margins[step].view_as(g_t)

      # Reach-avoid Bellman:
      #   non-terminal:  V*_t = min( g_t, max( l_t, γ·V_nx ) )
      #   terminal:      V*_t = g_t
      future_term = next_is_not_terminal * self.gamma * next_values
      reach = torch.maximum(ell_t, future_term)
      target = torch.where(
        next_is_not_terminal > 0,
        torch.minimum(g_t, reach),
        g_t,
      )
      delta = target - st.values[step]
      advantage = delta + self.gamma * self.lam * next_is_not_terminal * advantage
      st.returns[step] = advantage + st.values[step]

    st.advantages = st.returns - st.values
    if not self.normalize_advantage_per_mini_batch:
      st.advantages = (st.advantages - st.advantages.mean()) / (
        st.advantages.std() + 1e-8
      )
