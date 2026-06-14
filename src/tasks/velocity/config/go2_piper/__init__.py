from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_piper_flat_env_cfg,
  unitree_go2_piper_rough_env_cfg,
)
from .rl_cfg import unitree_go2_piper_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Piper-Rough",
  env_cfg=unitree_go2_piper_rough_env_cfg(),
  play_env_cfg=unitree_go2_piper_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_piper_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Piper-Flat",
  env_cfg=unitree_go2_piper_flat_env_cfg(),
  play_env_cfg=unitree_go2_piper_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_piper_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
