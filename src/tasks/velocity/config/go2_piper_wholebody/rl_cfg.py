"""RL configuration for Go2 + Piper whole-body control.

The action / observation spaces of this task are larger than the Go2-only
walking task (18-D action, larger obs because of EE pose + EE command), so
the network is sized up modestly. Single MLP for v1 — split-head policy
(per ManipLoco) and Advantage Mixing PPO are deferred until v1 is shown
to train.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_piper_wholebody_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
      noise_std_type="scalar",
    ),
    critic=RslRlModelCfg(
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
      learning_rate=2.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_piper_wholebody",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=15001,
  )
