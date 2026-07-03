"""Finetune the flat walker for higher speed (lin_vel_x up to 3.2 m/s).

Resumes from the stiffness-20 go2_velocity run's model_1000 and extends the
command range so the walker can deliver the momentum the crossing safety
policy's jump regime was trained at (2.2+ m/s). Bypasses tyro (negative tuple
values in --env.commands... aren't parseable on the CLI).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from train import TrainConfig, launch_training

TASK = "Unitree-Go2-Flat"

env_cfg = load_env_cfg(TASK)
agent_cfg = load_rl_cfg(TASK)

# Extended speed range (training range was (-1.0, 2.0)).
env_cfg.commands["twist"].ranges.lin_vel_x = (-1.0, 3.2)
env_cfg.scene.num_envs = 2048

# Faster gait clock: the fixed 0.6 s trot period caps speed at ~1.3 m/s
# (frequency-pinned; stride length is kinematically bounded). 0.4 s = 2.5 Hz.
# Reward and phase OBS must stay in sync.
env_cfg.rewards["foot_gait"].params["period"] = 0.3
for group in ("actor", "critic"):
  env_cfg.observations[group].terms["phase"].params["period"] = 0.3

agent_cfg.max_iterations = 2500
agent_cfg.resume = True
agent_cfg.load_run = "2026-07-02_15-09-26_fast_walker2"
agent_cfg.load_checkpoint = "model_7498.pt"
agent_cfg.run_name = "fast_walker3"

launch_training(task_id=TASK, args=TrainConfig(env=env_cfg, agent=agent_cfg))
