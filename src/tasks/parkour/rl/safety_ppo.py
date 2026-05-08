"""Safety PPO algorithm with Safety Bellman Backup.

Replaces standard GAE return computation with the safety Bellman update:
    v_to_go = min(g_t, v_next)
where g_t is the safety margin (positive = safe, negative = unsafe).

Reference: Fisac et al., "Bridging Hamilton-Jacobi Safety Analysis and
Reinforcement Learning", ICRA 2019.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class SafetyPPO(PPO):
  """PPO with Safety Bellman Backup for return computation.

  The only differences from standard PPO are:
  1. ``compute_returns`` uses ``min(g_t, v_next)`` instead of ``r + γ·v_next``.
  2. ``process_env_step`` skips the standard timeout bootstrapping
     (adding γ·V to the reward), which would corrupt the safety margin.
  """

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record a transition without timeout bootstrapping.

    Standard PPO adds ``γ * V(s)`` to the reward at timeouts so that
    the truncated return approximates the infinite-horizon return.  For
    safety RL the reward IS the safety margin g(s) — an absolute
    physical quantity — and adding a value estimate would destroy its
    semantics.  Timeouts are simply treated as terminal states with the
    last observed g(s) as the return.
    """
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      assert self.transition.rewards is not None
      self.transition.rewards += self.intrinsic_rewards

    # NOTE: intentionally skip timeout bootstrapping.

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute returns using the Safety Bellman Backup.

    Standard GAE:
        δ_t = r_t + γ·(1-d)·V(s_{t+1}) - V(s_t)

    Safety Bellman:
        v_to_go = min(g_t, V(s_{t+1}))
        target  = (1 - γ·(1-d))·g_t + γ·(1-d)·v_to_go
        δ_t     = target - V(s_t)

    At terminal states (d=1) the target collapses to g_t.  At
    non-terminal states it interpolates between the immediate margin
    and the worst-case future margin.
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
      v_to_go = torch.minimum(g_t, next_values)

      target = (
        1.0 - self.gamma * next_is_not_terminal
      ) * g_t + self.gamma * next_is_not_terminal * v_to_go
      delta = target - st.values[step]
      advantage = delta + self.gamma * self.lam * next_is_not_terminal * advantage
      st.returns[step] = advantage + st.values[step]

    st.advantages = st.returns - st.values
    if not self.normalize_advantage_per_mini_batch:
      st.advantages = (st.advantages - st.advantages.mean()) / (
        st.advantages.std() + 1e-8
      )
