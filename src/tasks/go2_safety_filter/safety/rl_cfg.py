"""RL configuration for the Go2 safety (shield fallback) policy.

Identical to the upstream parkour safety runner cfg — CNN actor over depth +
proprioception, MLP critic over privileged obs, ``SafetyPPO`` algorithm with
the Safety Bellman Backup. Exposed from this package for symmetry with the
walking config.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_safety_shield_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
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
      class_name="src.tasks.parkour.rl.safety_ppo.SafetyPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
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
      "actor": ("proprioception", "depth"),
      "critic": ("critic",),
    },
    experiment_name="go2_safety_shield",
    save_interval=100,
    num_steps_per_env=48,
    max_iterations=200_000,
  )


def unitree_go2_reach_avoid_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reach-avoid runner cfg — **hardware-deployable** observation space.

  Uses :class:`ReachAvoidPPO` with the reach-avoid Bellman backup
  ``V = min(g, max(l, γ·V'))``.  The actor is a CNN over the ``proprioception``
  group (no raycast ``height_scan``) + the ``depth`` image (RGBD), matching the
  real Go2's sensing (proprioception + depth camera).  The raycast heightfield
  and other privileged signals live only in the ``critic`` group.  The
  safety/reach margins ``g``/``l`` are computed in the wrapper from sim state
  (raycast + base velocity) — that is the training reward signal, not a policy
  observation, so it does not affect deployability.
  """
  return RslRlOnPolicyRunnerCfg(
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
      # Low entropy bonus: with the safety-Bellman reward (mean ~0) a larger
      # bonus dominates the policy gradient and blows up the action log-std
      # (over-jumping / flipping). The runner additionally clamps log_std.
      entropy_coef=0.0005,
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
      "actor": ("proprioception", "depth"),
      "critic": ("critic",),
    },
    experiment_name="go2_reach_avoid",
    save_interval=100,
    num_steps_per_env=48,
    max_iterations=60_000,
  )
