"""RL config for on-policy ISAACS (two-player adversarial reach-avoid).

Same ctrl player as the crossing-chain task (MLP over proprioception,
ReachAvoidPPO — warm-starts model_28799).  Adds an ``isaacs`` config block the
:class:`Go2IsaacsOnPolicyRunner` reads for the min player and phase machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.go2_safety_filter.crossing_chain.rl_cfg import (
  unitree_go2_crossing_chain_ppo_runner_cfg,
)


@dataclass
class RslRlIsaacsRunnerCfg(RslRlOnPolicyRunnerCfg):
  isaacs: dict = field(default_factory=dict)


def unitree_go2_crossing_chain_isaacs_runner_cfg() -> RslRlIsaacsRunnerCfg:
  base = unitree_go2_crossing_chain_ppo_runner_cfg()
  cfg = RslRlIsaacsRunnerCfg(
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
    obs_groups=base.obs_groups,
    experiment_name="go2_crossing_chain_isaacs",
    save_interval=200,
    num_steps_per_env=48,
    max_iterations=6000,
    isaacs={
      "force_max": 50.0,
      "dstb_pretrain_iters": 400,
      "ctrl_iters_per_cycle": 12,
      "dstb_iters_per_cycle": 3,
      "force_scale_ramp_iters": 200,
      "rest_edge_clearance": 0.3,
      "edge_ramp_ctrl_iters": 300,
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
  return cfg
