"""RL config for the crossing-CHAIN task.

Same MLP-over-proprioception actor/critic as the single-gap crossing (so it can
WARM-START from that converged checkpoint, ``model_4000``), but the algorithm is
``ReachAvoidPPO`` (Bellman backup ``V = min(g, max(l, γ·V'))``) instead of
avoid-only ``SafetyPPO`` — the reach term ``l`` (forward progress) supplies the
liveness drive that makes the robot keep running between gaps instead of stopping
after the first landing.  Paired with ``ParkourReachAvoidOnPolicyRunner`` (which
re-wraps the env with the g+l wrapper and clamps the actor log-std).
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2_crossing_chain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      # Low entropy bonus (mean-~0 safety reward); the runner also clamps log_std.
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
    experiment_name="go2_crossing_chain",
    save_interval=200,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
