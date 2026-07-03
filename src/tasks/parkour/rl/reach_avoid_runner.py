"""Reach-avoid on-policy runner for Go2 parkour.

Swaps the standard ``RslRlVecEnvWrapper`` for
``ParkourReachAvoidVecEnvWrapper`` so the training signal is the safety margin
``g(s)`` (reward) plus the reach margin ``l(s)`` (``extras["target_margin"]``),
consumed by :class:`ReachAvoidPPO`.

Also installs a **log-std clamp** on the actor.  Under the safety/reach-avoid
Bellman backup the per-step reward ``g(s)`` has mean ≈ 0, so the entropy bonus
dominates the policy gradient and pushes ``actor.log_std`` up without bound
(observed: std 0.3 → 12+ over a few thousand iters), collapsing the policy into
huge-variance noise — the robot flails, over-jumps, and flips.  Capping
``log_std`` in ``[log 0.10, log 1.0]`` after every optimizer step keeps
exploration bounded (and NaN-sanitizes grads/params for robustness).  Mirrors
``MjlabSafety_CLINC``'s ``_install_log_std_clamp``.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.env import VecEnv

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from src.tasks.parkour.rl.reach_avoid_vecenv_wrapper import (
  ParkourReachAvoidVecEnvWrapper,
)
from src.tasks.parkour.rl.runner import ParkourOnPolicyRunner

# Per-dimension log-std cap.  Upper = log(0.40): std=1.0 (log 0) was still far
# too noisy and the policy railed to it and collapsed into tipping, so we cap
# exploration much lower.  Lower = log(0.10) keeps the policy stochastic enough
# to keep exploring rather than collapsing to a deterministic freeze.
_LOG_STD_MAX = math.log(0.40)
_LOG_STD_MIN = math.log(0.10)


def _install_log_std_clamp(runner: ParkourOnPolicyRunner) -> None:
  """Keep ``actor.log_std`` in ``[_LOG_STD_MIN, _LOG_STD_MAX]`` and finite after
  every gradient step.  No-op if the actor doesn't expose ``log_std``."""
  actor = runner.alg.actor
  if not hasattr(actor, "log_std"):
    return

  optimizer = runner.alg.optimizer

  def _pre_step_sanitize_grads(_optimizer, *args, **kwargs) -> None:
    # Zero NaN/Inf grads before Adam corrupts its running statistics.
    with torch.no_grad():
      for group in _optimizer.param_groups:
        for p in group["params"]:
          if p.grad is None:
            continue
          if not torch.isfinite(p.grad).all():
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

  def _post_step_sanitize(_optimizer, *args, **kwargs) -> None:
    with torch.no_grad():
      ls = actor.log_std.data
      torch.nan_to_num_(ls, nan=_LOG_STD_MAX, posinf=_LOG_STD_MAX, neginf=_LOG_STD_MIN)
      ls.clamp_(min=_LOG_STD_MIN, max=_LOG_STD_MAX)

  optimizer.register_step_pre_hook(_pre_step_sanitize_grads)
  optimizer.register_step_post_hook(_post_step_sanitize)
  _post_step_sanitize(optimizer)  # sanitize once now


class ParkourReachAvoidOnPolicyRunner(ParkourOnPolicyRunner):
  """Runner that re-wraps the env with the reach-avoid wrapper.

  ``train.py`` always wraps the environment with ``RslRlVecEnvWrapper``.  This
  runner detects the standard wrapper and re-wraps the inner environment with
  ``ParkourReachAvoidVecEnvWrapper`` so that ``g(s)`` / ``l(s)`` drive training,
  then installs the log-std clamp.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if isinstance(env, RslRlVecEnvWrapper) and not isinstance(
      env, ParkourReachAvoidVecEnvWrapper
    ):
      env = ParkourReachAvoidVecEnvWrapper(
        env.env,
        clip_actions=env.clip_actions,
      )
    super().__init__(env, train_cfg, log_dir, device)
    _install_log_std_clamp(self)


class ParkourReachRestOnPolicyRunner(ParkourOnPolicyRunner):
  """Reach-avoid runner with the SAFETY-FILTER 'rest' target.

  Same as :class:`ParkourReachAvoidOnPolicyRunner` but the reach margin ``l`` is
  "come to a safe stop" (``reach_mode='rest'``) rather than velocity-command
  liveness — so braking / jumping / chaining all emerge as instrumental ways to
  reach safe rest given the arrival momentum (the deployment safety-filter
  objective).  A mild bias credits resting further along (crossing when safe).
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if isinstance(env, RslRlVecEnvWrapper) and not isinstance(
      env, ParkourReachAvoidVecEnvWrapper
    ):
      env = ParkourReachAvoidVecEnvWrapper(
        env.env,
        clip_actions=env.clip_actions,
        reach_mode="rest",
        v_rest=0.3,
        v_rest_norm=0.5,
        cross_bias_weight=0.3,
        cross_bias_scale=3.0,
      )
    super().__init__(env, train_cfg, log_dir, device)
    _install_log_std_clamp(self)


class GapReachAvoidOnPolicyRunner(ParkourOnPolicyRunner):
  """Reach-avoid runner for the gaps-only task (foothold reach margin)."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    from src.tasks.parkour.rl.gap_reach_avoid_vecenv_wrapper import (
      GapReachAvoidVecEnvWrapper,
    )

    if isinstance(env, RslRlVecEnvWrapper) and not isinstance(
      env, GapReachAvoidVecEnvWrapper
    ):
      env = GapReachAvoidVecEnvWrapper(env.env, clip_actions=env.clip_actions)
    super().__init__(env, train_cfg, log_dir, device)
    _install_log_std_clamp(self)
