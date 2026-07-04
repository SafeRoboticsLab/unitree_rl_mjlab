"""RL config for on-policy ISAACS on the crawl task (two-player adversarial).

Same isaacs block as the validated crossing-chain league run (force 50 N,
pretrain 400, 3 dstb / 12 ctrl, per-env survival force scale), pointed at the
crawl ctrl player.  ``rest_edge_clearance`` stays 0 — the crawl terrain has no
edges; the bar-specific rest robustification is the obstacle window, which is
always on via the env cfg.
"""

from __future__ import annotations

from src.tasks.go2_safety_filter.crawl.rl_cfg import unitree_go2_crawl_ppo_runner_cfg
from src.tasks.go2_safety_filter.crossing_chain.isaacs_rl_cfg import (
  RslRlIsaacsRunnerCfg,
)


def unitree_go2_crawl_isaacs_runner_cfg() -> RslRlIsaacsRunnerCfg:
  base = unitree_go2_crawl_ppo_runner_cfg()
  return RslRlIsaacsRunnerCfg(
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
    obs_groups=base.obs_groups,
    experiment_name="go2_crawl_isaacs",
    save_interval=200,
    num_steps_per_env=48,
    max_iterations=6000,
    isaacs={
      "force_max": 50.0,
      "dstb_pretrain_iters": 400,
      "ctrl_iters_per_cycle": 12,
      "dstb_iters_per_cycle": 3,
      "force_scale_ramp_iters": 200,
      "rest_edge_clearance": 0.0,
      "edge_ramp_ctrl_iters": 0,
      "dstb_actor": {
        "hidden_dims": (256, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
        "init_noise_std": 0.5,
        "noise_std_type": "log",
      },
      "dstb_algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.002,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3.0e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
      },
    },
  )
