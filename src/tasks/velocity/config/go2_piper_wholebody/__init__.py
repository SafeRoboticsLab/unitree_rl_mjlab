from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_piper_wholebody_flat_env_cfg,
  unitree_go2_piper_wholebody_rough_env_cfg,
)
from .rl_cfg import unitree_go2_piper_wholebody_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Piper-WholeBody-Rough",
  env_cfg=unitree_go2_piper_wholebody_rough_env_cfg(),
  play_env_cfg=unitree_go2_piper_wholebody_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_piper_wholebody_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Piper-WholeBody-Flat",
  env_cfg=unitree_go2_piper_wholebody_flat_env_cfg(),
  play_env_cfg=unitree_go2_piper_wholebody_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_piper_wholebody_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
