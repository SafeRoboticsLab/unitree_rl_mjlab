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
  GapReachAvoidOnPolicyRunner,
  ParkourOnPolicyRunner,
  ParkourReachAvoidOnPolicyRunner,
  ParkourReachRestOnPolicyRunner,
  ParkourSafetyOnPolicyRunner,
)

from .crossing_chain.env_cfg import unitree_go2_crossing_chain_env_cfg
from .crossing_chain.rl_cfg import unitree_go2_crossing_chain_ppo_runner_cfg

from .walking.env_cfg import unitree_go2_walking_env_cfg
from .walking.rl_cfg import unitree_go2_walking_ppo_runner_cfg
from .safety.env_cfg import unitree_go2_safety_shield_env_cfg
from .safety.reach_avoid_env_cfg import unitree_go2_reach_avoid_env_cfg
from .safety.rl_cfg import (
  unitree_go2_reach_avoid_ppo_runner_cfg,
  unitree_go2_safety_shield_ppo_runner_cfg,
)
from .gap.env_cfg import unitree_go2_gap_reach_avoid_env_cfg
from .gap.rl_cfg import unitree_go2_gap_reach_avoid_ppo_runner_cfg
from .landing.env_cfg import unitree_go2_landing_env_cfg
from .landing.rl_cfg import unitree_go2_landing_ppo_runner_cfg
from .crossing.env_cfg import unitree_go2_crossing_env_cfg
from .crossing.rl_cfg import unitree_go2_crossing_ppo_runner_cfg
from .crossing.adv_env_cfg import unitree_go2_crossing_adv_env_cfg
from .gauntlet.env_cfg import unitree_go2_gauntlet_env_cfg

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

# Reach-avoid safety policy: crosses jumpable gaps / tracks the velocity
# command when safe, falls back to pure safety otherwise.  Shares the safety
# shield env (parkour terrain + raycast); uses the reach-avoid runner + algo.
register_mjlab_task(
  task_id="Unitree-Go2-Reach-Avoid",
  env_cfg=unitree_go2_reach_avoid_env_cfg(),
  play_env_cfg=unitree_go2_reach_avoid_env_cfg(play=True),
  rl_cfg=unitree_go2_reach_avoid_ppo_runner_cfg(),
  runner_cls=ParkourReachAvoidOnPolicyRunner,
)

# Minimal gaps-only reach-avoid: stop before uncrossable gaps, jump small ones,
# stop/continue once across. Privileged proprioception actor (incl. height_scan).
register_mjlab_task(
  task_id="Unitree-Go2-Gap-ReachAvoid",
  env_cfg=unitree_go2_gap_reach_avoid_env_cfg(),
  play_env_cfg=unitree_go2_gap_reach_avoid_env_cfg(play=True),
  rl_cfg=unitree_go2_gap_reach_avoid_ppo_runner_cfg(),
  runner_cls=GapReachAvoidOnPolicyRunner,
)

# Landing sub-task: mid-air-over-gap spawn -> learn soft-landing. Avoid-only
# SafetyPPO (g only), meant for very large num_envs to surface the rare win.
register_mjlab_task(
  task_id="Unitree-Go2-Landing",
  env_cfg=unitree_go2_landing_env_cfg(),
  play_env_cfg=unitree_go2_landing_env_cfg(play=True),
  rl_cfg=unitree_go2_landing_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)

# Reverse-curriculum crossing: extend backward from the learned landing to the
# launch (per-env back_level promotes as it reaches the far platform). SafetyPPO.
register_mjlab_task(
  task_id="Unitree-Go2-Crossing",
  env_cfg=unitree_go2_crossing_env_cfg(),
  play_env_cfg=unitree_go2_crossing_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)

# Crossing + adversary (increment 1): launch-cross-land must survive a curriculum
# push. Reuses the crossing rl_cfg (same experiment_name) so it can warm-start
# from the converged crossing checkpoint.
register_mjlab_task(
  task_id="Unitree-Go2-Crossing-Adv",
  env_cfg=unitree_go2_crossing_adv_env_cfg(),
  play_env_cfg=unitree_go2_crossing_adv_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)

# Safety-filter crossing: arrive with random momentum, reach a SAFE STOP. The
# reach term l = "come to rest" (rest mode), so braking / one jump / chaining a
# cluster all emerge as instrumental ways to reach safe rest given the arrival
# momentum (deployment safety-filter objective); a mild bias crosses when safe.
# Warm-starts from the single-gap crossing (model_4000): same MLP-proprioception
# actor + ReachAvoidPPO. Safety-filter terrain + survive-based curriculum.
register_mjlab_task(
  task_id="Unitree-Go2-Crossing-Chain",
  env_cfg=unitree_go2_crossing_chain_env_cfg(),
  play_env_cfg=unitree_go2_crossing_chain_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_chain_ppo_runner_cfg(),
  runner_cls=ParkourReachRestOnPolicyRunner,
)

# Gauntlet eval: progressive gaps-grow/platforms-shrink track.
register_mjlab_task(
  task_id="Unitree-Go2-Gauntlet",
  env_cfg=unitree_go2_gauntlet_env_cfg(),
  play_env_cfg=unitree_go2_gauntlet_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)
