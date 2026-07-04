"""Disturbance (min-player) PPO for the on-policy ISAACS game.

The zero-sum reach-avoid game optimizes ONE value

    V(s) = min( g(s), max( l(s), gamma * V(s') ) )

with the control player ascending it and the disturbance player descending it.
On-policy, both players can share the exact same targets/returns computed by
:class:`ReachAvoidPPO.compute_returns`; the min player differs ONLY in the
sign of the advantage its PPO surrogate consumes (negation commutes with the
affine advantage normalization).  Its critic fits the same returns, giving the
min player its own value baseline without coupling the optimizers.

References: Hsu, Nguyen, Fisac, "ISAACS: Iterative Soft Adversarial
Actor-Critic for Safety", L4DC 2023; Pinto et al., "Robust Adversarial RL",
ICML 2017.
"""

from __future__ import annotations

from tensordict import TensorDict

from src.tasks.parkour.rl.reach_avoid_ppo import ReachAvoidPPO


class DstbReachAvoidPPO(ReachAvoidPPO):
  """Min-player: identical reach-avoid targets, negated advantage."""

  def compute_returns(self, obs: TensorDict) -> None:
    super().compute_returns(obs)
    # Descend the reach-avoid value: the PPO clip surrogate maximizes
    # E[ratio * A]; negating A makes this player minimize the ctrl objective.
    self.storage.advantages.neg_()
