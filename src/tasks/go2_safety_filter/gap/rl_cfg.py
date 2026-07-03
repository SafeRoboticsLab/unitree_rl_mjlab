"""RL config for the minimal gaps-only reach-avoid task.

Privileged-actor first pass: an MLP over the ``proprioception`` group (which
includes the raycast ``height_scan``, so the gap geometry is directly
observable).  Uses ``ReachAvoidPPO`` with the foothold reach margin (supplied by
``GapReachAvoidVecEnvWrapper`` via the runner) and the log-std clamp + low
entropy that keep the action std from running away.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_gap_reach_avoid_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=0.3,
      noise_std_type="log",
    ),
    critic=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="src.tasks.parkour.rl.reach_avoid_ppo.ReachAvoidPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "actor": ("proprioception",),
      "critic": ("critic",),
    },
    experiment_name="go2_gap_reach_avoid",
    save_interval=200,
    num_steps_per_env=48,
    max_iterations=20_000,
  )
