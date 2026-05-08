"""Go2 safety-shield task stack.

Registers two tasks:

* ``Unitree-Go2-Walking`` — nominal MLP locomotion policy on rough terrain
  (no gaps, no crawl barriers). Standard PPO.
* ``Unitree-Go2-Safety-Shield`` — CNN safety policy on parkour terrain with
  mid-air-over-gap resets. SafetyPPO with Safety Bellman Backup.

Both policies share an identical ``proprioception`` observation group so they
can be combined at play time by :class:`ShieldedPolicy` without any obs
adapter. See ``DESIGN.md`` in this package for the full rationale.
"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.parkour.rl import (
  ParkourOnPolicyRunner,
  ParkourSafetyOnPolicyRunner,
)

from .walking.env_cfg import unitree_go2_walking_env_cfg
from .walking.rl_cfg import unitree_go2_walking_ppo_runner_cfg
from .safety.env_cfg import unitree_go2_safety_shield_env_cfg
from .safety.rl_cfg import unitree_go2_safety_shield_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Walking",
  env_cfg=unitree_go2_walking_env_cfg(),
  play_env_cfg=unitree_go2_walking_env_cfg(play=True),
  rl_cfg=unitree_go2_walking_ppo_runner_cfg(),
  runner_cls=ParkourOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Safety-Shield",
  env_cfg=unitree_go2_safety_shield_env_cfg(),
  play_env_cfg=unitree_go2_safety_shield_env_cfg(play=True),
  rl_cfg=unitree_go2_safety_shield_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)
