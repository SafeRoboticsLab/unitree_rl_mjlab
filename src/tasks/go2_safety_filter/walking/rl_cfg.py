"""RL configuration for the Go2 walking (shield nominal) policy.

MLP actor over proprioception only. Critic uses the privileged ``critic``
group.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_walking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
      noise_std_type="scalar",
    ),
    critic=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "actor": ("proprioception",),
      "critic": ("critic",),
    },
    experiment_name="go2_walking_shield",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_001,
  )
