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
from src.tasks.parkour.rl.isaacs_runner import Go2IsaacsOnPolicyRunner

from .crossing_chain.env_cfg import (
  unitree_go2_crossing_chain_env_cfg,
  unitree_go2_crossing_chain_isaacs_env_cfg,
)
from .crossing_chain.adv_env_cfg import unitree_go2_crossing_chain_adv_env_cfg
from .crossing_chain.rl_cfg import unitree_go2_crossing_chain_ppo_runner_cfg
from .crossing_chain.isaacs_rl_cfg import unitree_go2_crossing_chain_isaacs_runner_cfg

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
from .crawl.env_cfg import (
  unitree_go2_crawl_env_cfg,
  unitree_go2_crawl_isaacs_env_cfg,
)
from .crawl.adv_env_cfg import unitree_go2_crawl_adv_env_cfg
from .crawl.rl_cfg import (
  unitree_go2_crawl_avoid_ppo_runner_cfg,
  unitree_go2_crawl_ppo_runner_cfg,
)
from .crawl.isaacs_rl_cfg import unitree_go2_crawl_isaacs_runner_cfg

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

# ISAACS increment 0: crossing-chain + sustained RANDOM base push with a per-env
# force curriculum (5 N steps up to 50 N = the game's disturbance bound). Same
# rl_cfg/experiment_name as the chain task so it warm-starts model_28799.
register_mjlab_task(
  task_id="Unitree-Go2-Crossing-Chain-Adv",
  env_cfg=unitree_go2_crossing_chain_adv_env_cfg(),
  play_env_cfg=unitree_go2_crossing_chain_adv_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_chain_ppo_runner_cfg(),
  runner_cls=ParkourReachRestOnPolicyRunner,
)

# ISAACS increments 1-2: two-player adversarial reach-avoid. Plain env cfg (no
# push event: the adversarial wrapper owns the wrench channel); reset_takeover
# gets edge_margin=0.3 so 'stoppable' matches the robustified rest set.
register_mjlab_task(
  task_id="Unitree-Go2-Crossing-Chain-ISAACS",
  env_cfg=unitree_go2_crossing_chain_isaacs_env_cfg(),
  play_env_cfg=unitree_go2_crossing_chain_isaacs_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_chain_isaacs_runner_cfg(),
  runner_cls=Go2IsaacsOnPolicyRunner,
)

# Gauntlet eval: progressive gaps-grow/platforms-shrink track.
register_mjlab_task(
  task_id="Unitree-Go2-Gauntlet",
  env_cfg=unitree_go2_gauntlet_env_cfg(),
  play_env_cfg=unitree_go2_gauntlet_env_cfg(play=True),
  rl_cfg=unitree_go2_crossing_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)

# Crawl safety filter (skill 2): keep moving forward under a bar, ducking as
# low as it takes; STOP if the bar is below the crouch feasibility floor.
# FRESH policy (actor natively includes the forward/up bar-scan fans).
# ReachAvoidPPO with COMMAND / velocity-liveness reach (l = forward-speed
# liveness) + a forward HEIGHT curriculum (start high, lower a notch on
# each crossing). Command mode, not rest: the Go2 can brake out of any
# approach, so rest collapsed to stop-always.
register_mjlab_task(
  task_id="Unitree-Go2-Crawl",
  env_cfg=unitree_go2_crawl_env_cfg(),
  play_env_cfg=unitree_go2_crawl_env_cfg(play=True),
  rl_cfg=unitree_go2_crawl_ppo_runner_cfg(),
  runner_cls=ParkourReachAvoidOnPolicyRunner,
)

# Crawl ISAACS increment 0: + sustained RANDOM push curriculum (5 N steps to
# 50 N). Same rl_cfg/experiment_name as the crawl task -> in-place warm start
# of the phase-0 checkpoint.
register_mjlab_task(
  task_id="Unitree-Go2-Crawl-Adv",
  env_cfg=unitree_go2_crawl_adv_env_cfg(),
  play_env_cfg=unitree_go2_crawl_adv_env_cfg(play=True),
  rl_cfg=unitree_go2_crawl_ppo_runner_cfg(),
  runner_cls=ParkourReachAvoidOnPolicyRunner,
)

# Crawl ISAACS increments 1-2: two-player adversarial reach-avoid + league.
# Plain cfg (adversarial wrapper owns the wrench channel); stop_margin 0.3 and
# pinned curricula.
register_mjlab_task(
  task_id="Unitree-Go2-Crawl-ISAACS",
  env_cfg=unitree_go2_crawl_isaacs_env_cfg(),
  play_env_cfg=unitree_go2_crawl_isaacs_env_cfg(play=True),
  rl_cfg=unitree_go2_crawl_isaacs_runner_cfg(),
  runner_cls=Go2IsaacsOnPolicyRunner,
)

# Avoid-only baseline (SafetyPPO, g only): predicted stop-always at the bar —
# the motivating contrast row of the crawl benchmark.
register_mjlab_task(
  task_id="Unitree-Go2-Crawl-Avoid",
  env_cfg=unitree_go2_crawl_env_cfg(),
  play_env_cfg=unitree_go2_crawl_env_cfg(play=True),
  rl_cfg=unitree_go2_crawl_avoid_ppo_runner_cfg(),
  runner_cls=ParkourSafetyOnPolicyRunner,
)
