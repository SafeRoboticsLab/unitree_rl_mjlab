"""RL configuration for the Go2 parkour **safety** task.

Uses :class:`SafetyPPO` (which implements the Safety Bellman Backup)
with hyperparameters adapted to the narrow-range safety margin signal:

* ``init_std``/``entropy_coef`` are lowered — with all-negative
  failures and small positive margins, large exploration noise drowns
  out the signal and can push the value function into a "death
  spiral".
* ``schedule="fixed"`` — adaptive KL interacts poorly with the sparse
  safety signal (low early-KL → LR up → noise up → more failures).
* ``num_steps_per_env`` is lengthened for more stable value targets.
* Actor model is the same CNN architecture as the standard parkour
  runner so the policy can still consume depth images.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_parkour_safety_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Go2 parkour safety training."""
  return RslRlOnPolicyRunnerCfg(
    # Actor: CNN over depth + proprioception (no raycast).
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
      # Lower initial std so exploration does not drown out the
      # narrow-range g(s) signal.  ``log`` parameterisation prevents
      # unbounded std growth while the critic learns.
      stochastic=True,
      init_noise_std=0.3,
      noise_std_type="log",
    ),
    # Critic: MLP with privileged observations (includes raycast scan).
    critic=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="src.tasks.parkour.rl.safety_ppo.SafetyPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,  # Lower: avoid noise explosion under negative reward.
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="fixed",  # Adaptive KL interacts poorly with safety signal.
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # Map observation groups from env to model inputs.
    # Actor sees depth + proprioception; critic consumes the privileged
    # group (which still contains the raycast scan).
    obs_groups={
      "actor": ("proprioception", "depth"),
      "critic": ("critic",),
    },
    experiment_name="go2_parkour_safety",
    save_interval=100,
    num_steps_per_env=48,  # Longer rollout: more stable value targets.
    max_iterations=200_000,
  )
