"""Safety-aware on-policy runner for Go2 parkour tasks.

Swaps the standard ``RslRlVecEnvWrapper`` for ``ParkourSafetyVecEnvWrapper``
so that the training reward is the safety margin g(s) instead of the
shaped parkour reward sum.
"""

from __future__ import annotations

from rsl_rl.env import VecEnv

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from src.tasks.parkour.rl.runner import ParkourOnPolicyRunner
from src.tasks.parkour.rl.safety_vecenv_wrapper import ParkourSafetyVecEnvWrapper


class ParkourSafetyOnPolicyRunner(ParkourOnPolicyRunner):
  """Runner that replaces the env wrapper with a safety wrapper.

  ``train.py`` always wraps the environment with ``RslRlVecEnvWrapper``.
  This runner detects the standard wrapper and re-wraps the inner
  environment with ``ParkourSafetyVecEnvWrapper`` so that g(s) is used
  as the training signal.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if isinstance(env, RslRlVecEnvWrapper) and not isinstance(
      env, ParkourSafetyVecEnvWrapper
    ):
      env = ParkourSafetyVecEnvWrapper(
        env.env,
        clip_actions=env.clip_actions,
      )
    super().__init__(env, train_cfg, log_dir, device)
