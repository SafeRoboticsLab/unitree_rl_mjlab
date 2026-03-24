"""RL configuration for Unitree Go2 parkour task.

Uses CNNModel for the actor (processes depth images + proprioception)
and MLPModel for the critic (privileged proprioceptive info only).
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_parkour_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Go2 parkour with vision."""
  return RslRlOnPolicyRunnerCfg(
    # Actor: CNNModel processes depth (2D) + proprioception (1D).
    actor=RslRlModelCfg(
      class_name="CNNModel",
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg={
        "output_channels": (32, 64, 64),
        "kernel_size": (5, 3, 3),
        "stride": (2, 2, 2),
        "padding": "zeros",
        "activation": "elu",
        "norm": "none",
        "max_pool": False,
        "global_pool": "avg",
      },
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    # Critic: MLP with privileged observations (no depth needed).
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
    # Map observation groups from env to model inputs.
    obs_groups={
      "actor": ("proprioception", "depth"),
      "critic": ("critic",),
    },
    experiment_name="go2_parkour",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=15001,
  )
