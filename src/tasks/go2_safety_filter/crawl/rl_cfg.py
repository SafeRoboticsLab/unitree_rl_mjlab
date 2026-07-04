"""RL configs for the crawl-under-low-bar task.

FRESH policy (user decision): the actor is built with the bar-scan dims
natively (proprioception gains two 7-ray forward fans), so no checkpoint
surgery and no warm start.  Hyperparameters are the crossing-chain settings
verbatim — same algorithm family (ReachAvoidPPO + rest objective), same
scale of problem.

Also exports the avoid-only SafetyPPO baseline cfg (predicted stop-always —
the motivating contrast for the benchmark).
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2_crawl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      entropy_coef=0.0005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("proprioception",), "critic": ("critic",)},
    experiment_name="go2_crawl",
    save_interval=200,
    num_steps_per_env=48,
    max_iterations=12_000,
  )


def unitree_go2_crawl_avoid_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Avoid-only baseline: same nets, SafetyPPO (g only, no liveness)."""
  cfg = unitree_go2_crawl_ppo_runner_cfg()
  cfg.algorithm.class_name = "src.tasks.parkour.rl.safety_ppo.SafetyPPO"
  cfg.experiment_name = "go2_crawl_avoid"
  return cfg
