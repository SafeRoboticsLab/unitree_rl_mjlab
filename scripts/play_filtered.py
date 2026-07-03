"""Interactive value-based safety filter: walking policy + crossing safety policy.

Loads two checkpoints:

* ``--walk-checkpoint`` — the TASK policy: blind flat-terrain walker
  (``Unitree-Go2-Flat``, 47-dim proprioception, knows nothing about gaps).
* ``--safe-checkpoint`` — the SAFETY policy: crossing-chain reach-avoid policy
  (``Unitree-Go2-Crossing-Chain``, proprioception + height_scan) and its
  SafetyPPO/ReachAvoidPPO CRITIC V(s).

Value-based filter (as in safety-stable-baselines): at every step, evaluate the
safety value V(s) with the safety critic.  If ``V(s) > epsilon`` the state is
safe -> execute the task (walking) action; else the safety policy overrides
(jump if momentum demands it, brake/stop otherwise).

The two policies have DIFFERENT observation layouts.  This script builds ONE
env (the gap world) that emits both: the walking policy's exact ``actor`` obs
group is grafted from the Flat task cfg, while the safety policy/critic consume
the chain env's own ``proprioception``/``critic`` groups.  The safety policy
and critic were trained with a CONSTANT twist command obs (1.0, 0, 0) and an
always-running gait clock, so their groups get pinned command/phase terms — the
real (teleop) command is only visible to the walking policy.

KNOWN LIMITATION (measured): handovers from MID-GAIT walking states are outside
the safety policy's training distribution (its takeover spawns used neutral
stance + velocity), so an engaged override fails (face-plant) on a large
fraction of mid-trot handovers.  The caution band + V smoothing mitigate but do
not close this; the fix is to finetune the safety policy with walking-gait
handover states in its spawn distribution.  See the research notes.

Teleop (native viewer window):
    I / K : forward / backward      J / L : turn left / right
    U / O : strafe left / right     0     : stop (zero command)
    T     : toggle episode terminations on/off (off = no auto-reset on falls)
    ENTER : force reset a fresh episode (built into the viewer)

Usage
-----
    python scripts/play_filtered.py \\
        --walk-checkpoint logs/rsl_rl/go2_velocity/2026-03-18_23-30-35/model_1000.pt \\
        --safe-checkpoint logs/rsl_rl/go2_crossing_chain/2026-07-01_22-21-40_chain_gated2/model_17999.pt \\
        --epsilon 0.05 --num-envs 1 --level 3
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import tyro
from tensordict import TensorDict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

WALK_TASK = "Unitree-Go2-Flat"
SAFE_TASK = "Unitree-Go2-Crossing-Chain"
# The command obs the safety policy/critic saw throughout training (constant).
# NOTE: the chain task inherited the gap cfg's twist (1.0, 1.0) — verified from
# the live env obs (command block = 1.0), NOT the 2.5 the old speed-l cfg used.
SAFE_CMD = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class FilteredPlayConfig:
  walk_checkpoint: str = (
    "logs/rsl_rl/go2_velocity/2026-07-02_15-09-26_fast_walker2/model_7498.pt"
  )
  walk_phase_period: float = 0.4
  """Gait-clock period (s) the walking checkpoint was trained with. The
  fast_walker2 line uses 0.4; the original go2_velocity checkpoints used 0.6."""
  safe_checkpoint: str = (
    "logs/rsl_rl/go2_crossing_chain/2026-07-02_23-10-03_chain_handover3/model_28799.pt"
  )
  epsilon: float = 0.25
  """Safety threshold: override with the safety policy when V(s) <= epsilon."""
  caution: float = 0.45
  """Caution threshold (> epsilon): when V <= caution, veto the user command
  (walking policy gets a zero command and slows itself — in-distribution) before
  escalating to the full safety override at epsilon."""
  hysteresis: float = 0.15
  """Release the override only when V > epsilon + hysteresis (and near rest)."""
  rest_speed: float = 0.4
  """Release the override only when base speed (m/s) is below this (safe stop)."""
  arm_distance: float | None = None
  """If set, the filter stays DORMANT (pure walking policy, full teleop) until
  the robot is within this distance (m) of the gap cluster — lets you build up
  momentum at full speed before the walking->filtering sequence kicks in.
  None = filter always active (previous behavior)."""
  armed_epsilon: float = 0.20
  """Engage threshold used INSIDE the armed zone (>= epsilon). The arm gate
  already blocks cruise-phase false positives, so a more conservative (higher)
  threshold is safe there and engages the jump with more runway."""
  instant: bool = False
  """Hard immediate switching: use RAW V (no median smoothing) and DISABLE the
  caution band (no command veto). The walking policy runs at full momentum
  until the very step V crosses the threshold, then the safety policy takes
  over abruptly — maximum momentum carried into the jump. The latch still
  holds the override until the maneuver completes."""
  num_envs: int = 1
  level: int | None = 3
  """Pin all envs to this terrain difficulty row (None = spread over rows)."""
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  log_every: int = 25
  """Print filter telemetry every N steps."""
  no_filter: bool = False
  """Disable the safety filter (pure walking policy) — for the baseline demo."""
  no_terminations: bool = False
  """Start with episode terminations disabled (toggle at runtime with the T
  key). While disabled the robot never auto-resets on falls/timeouts — useful
  for watching a failure play out; press ENTER to force a fresh episode."""


# --- Obs plumbing ----------------------------------------------------------


def _pinned_command(env, asset_cfg=None):
  """Constant command obs for the safety policy/critic (their training value)."""
  n = env.num_envs
  t = torch.tensor(SAFE_CMD, device=env.device, dtype=torch.float32)
  return t.unsqueeze(0).expand(n, -1).clone()


def _pinned_phase(env, period: float = 0.8, command_name: str = "twist"):
  """Gait clock WITHOUT the stand-mask.

  The parkour ``phase`` obs zeroes itself when the command norm < 0.1 — but the
  safety policy trained with a constant 2.5 command, so it has NEVER seen
  phase=(0,0); feeding it the masked phase (teleop cmd ~0) is OOD and it falls
  over from standing.  Reproduce the unmasked clock it always saw.
  """
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  out = torch.zeros(env.num_envs, 2, device=env.device)
  out[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  out[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  return out


def _reset_standing(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn standing near the start of the approach, zero velocity (teleop)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  pose = torch.stack(
    [0.4 + u(-0.1, 0.1), u(-0.3, 0.3), 0.05 + u(-0.01, 0.01),
     u(-0.02, 0.02), u(-0.02, 0.02), u(-0.05, 0.05)], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


def build_env_cfg(num_envs: int, walk_phase_period: float = 0.4):
  """Chain (gap-world) env cfg emitting BOTH policies' observation groups."""
  cfg = load_env_cfg(SAFE_TASK, play=True)
  cfg.scene.num_envs = num_envs

  # Long episodes; standing teleop spawn; no strata, no curriculum.
  cfg.episode_length_s = 120.0
  cfg.events["reset_base"] = EventTermCfg(func=_reset_standing, mode="reset", params={})
  cfg.curriculum = {}
  # CRITICAL: the inherited randomize_terrain event re-rolls each env's terrain
  # level AND origin on every reset — it would silently undo the --level pin
  # and desync the gap-distance reference used by --arm-distance.
  cfg.events.pop("randomize_terrain", None)

  # Teleop owns the command: stop the manager from resampling/standing/heading.
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False

  # Graft the walking policy's EXACT actor obs group (from the Flat task cfg)
  # as a new group named "actor" — the walking runner's obs_groups select it.
  walk_cfg = load_env_cfg(WALK_TASK, play=True)
  walk_group = copy.deepcopy(walk_cfg.observations["actor"])
  assert "height_scan" not in walk_group.terms, "expected blind Flat actor group"
  walk_group.terms["phase"].params["period"] = walk_phase_period
  cfg.observations["actor"] = walk_group

  # Pin the command + phase obs for the safety policy + critic to what they saw
  # in training (constant command; unmasked gait clock).  Teleop values / the
  # stand-masked phase would be OOD for their normalizers and policies.
  for gname in ("proprioception", "critic"):
    term = copy.deepcopy(cfg.observations[gname].terms["command"])
    term.func = _pinned_command
    term.params = {}
    cfg.observations[gname].terms["command"] = term

    phase_term = copy.deepcopy(cfg.observations[gname].terms["phase"])
    period = float(phase_term.params.get("period", 0.8))
    phase_term.func = _pinned_phase
    phase_term.params = {"period": period}
    cfg.observations[gname].terms["phase"] = phase_term

  return cfg


# --- Filter policy ----------------------------------------------------------


class ValueFilteredPolicy:
  """Value filter with hysteresis (latched override).

  Engage the safety policy when ``V(s) <= epsilon``.  Once engaged, STAY
  engaged until the maneuver completes — the robot is back in a clearly-safe
  state (``V > epsilon + hysteresis``) AND nearly at rest (the safety policy's
  own terminal condition).  Without the latch, V-noise flaps control between
  the two policies mid-maneuver and neither completes its action.
  """

  def __init__(
    self,
    walk_actor: Callable,
    safe_actor: Callable,
    safe_critic: Callable,
    epsilon: float,
    teleop_cmd: torch.Tensor,
    command_setter: Callable[[torch.Tensor], None],
    speed_getter: Callable[[], torch.Tensor],
    dones_getter: Callable[[], torch.Tensor],
    caution: float = 0.45,
    hysteresis: float = 0.15,
    rest_speed: float = 0.4,
    gap_dist_getter: Callable[[], torch.Tensor] | None = None,
    arm_distance: float | None = None,
    armed_epsilon: float = 0.20,
    instant: bool = False,
    log_fn: Callable[[str], None] | None = None,
    log_every: int = 25,
    disabled: bool = False,
  ) -> None:
    self._walk = walk_actor
    self._safe = safe_actor
    self._critic = safe_critic
    self._eps = float(epsilon)
    self._caution = float(caution)
    self._hys = float(hysteresis)
    self._rest_speed = float(rest_speed)
    self._teleop = teleop_cmd
    self._set_cmd = command_setter
    self._speed = speed_getter
    self._dones = dones_getter
    self._gap_dist = gap_dist_getter
    self._arm_distance = arm_distance
    self._armed_eps = max(float(armed_epsilon), float(epsilon))
    self._instant = bool(instant)
    self._log = log_fn
    self._log_every = log_every
    self._disabled = disabled
    self._i = 0
    self._engaged: torch.Tensor | None = None
    self._v_hist: torch.Tensor | None = None
    self._was_armed = False
    self._override_ema = 0.0

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    v_raw = self._critic(obs).squeeze(-1)  # (num_envs,)
    # V is noisy on the walking policy's (off-distribution) gait states —
    # transient dips of 1-3 steps punch below any threshold while cruising.
    # Median-of-last-5 smooths those out for the band logic; a RAW hard floor
    # (V <= 0) still reacts instantly to genuine emergencies.
    if self._v_hist is None:
      self._v_hist = v_raw.unsqueeze(0).repeat(5, 1)
    self._v_hist = torch.cat([self._v_hist[1:], v_raw.unsqueeze(0)], dim=0)
    fresh = self._dones().bool()
    if bool(fresh.any()):
      self._v_hist[:, fresh] = v_raw[fresh].unsqueeze(0)
    v_safe = v_raw if self._instant else self._v_hist.median(dim=0).values

    if self._engaged is None:
      self._engaged = torch.zeros_like(v_safe, dtype=torch.bool)
    # Fresh episodes start disengaged.
    self._engaged &= ~fresh

    engage = (v_safe <= self._eps) | (v_raw <= self._eps - 0.15)
    release = (v_safe > self._eps + self._hys) & (self._speed() < self._rest_speed)

    # Arm-distance gate: while far from the gap the filter is DORMANT — pure
    # walking at full teleop so momentum can build.  Arming gates only NEW
    # engagements/caution; an already-latched override completes its maneuver.
    if self._arm_distance is not None and self._gap_dist is not None:
      d = self._gap_dist() - 0.25  # front of the body, not the base center
      armed = (d < self._arm_distance) & (d > -1.2)  # past-the-gap disarms too
      # Once armed the window to the edge is short (<0.5 s at speed): react on
      # RAW V — the median smoothing (there to reject cruise-phase dips) would
      # eat 2-3 steps of the window and the feet reach the void first.
      # In --instant mode use the user's epsilon directly (LATE, DEEP handover:
      # engaging early at stoppable distance triggers the safety policy's
      # weakest skill, braking from mid-trot; engaging at the lip with momentum
      # triggers its strongest, the jump). Otherwise use the armed threshold.
      thr = self._eps if self._instant else self._armed_eps
      engage = armed & ((v_raw <= thr) | (v_safe <= thr))
      if self._log is not None and bool(armed[0]) != self._was_armed:
        self._log(f"[filter] {'>>> ARMED' if bool(armed[0]) else '<<< disarmed'} "
                  f"(front dist to gap {float(d[0]):+.2f} m, V_raw {float(v_raw[0]):+.3f})")
        self._was_armed = bool(armed[0])
    else:
      armed = torch.ones_like(engage)

    self._engaged = (self._engaged | engage) & ~release
    unsafe = self._engaged.clone()
    caution = (v_safe <= self._caution) & ~unsafe & armed
    if self._instant:
      caution = torch.zeros_like(caution)
    if self._disabled:
      unsafe = torch.zeros_like(unsafe)
      caution = torch.zeros_like(caution)

    # Command routing: normal teleop when clearly safe; ZERO command in the
    # caution band (the walker slows itself — an in-distribution deceleration —
    # so a possible handover happens from a braking stance, not mid-trot).
    # The safety policy's command obs is pinned and unaffected.
    cmd = self._teleop.clone()
    if bool(caution[0]):  # env0 drives the (shared) teleop command
      cmd.zero_()
    self._set_cmd(cmd)

    a_walk = self._walk(obs)
    if bool(unsafe.any()):
      a_safe = self._safe(obs)
      action = torch.where(unsafe.unsqueeze(-1), a_safe, a_walk)
    else:
      action = a_walk

    self._override_ema = 0.98 * self._override_ema + 0.02 * float(
      unsafe.float().mean().item())
    if self._log is not None and self._i % self._log_every == 0:
      mode = ("SAFETY" if bool(unsafe[0])
              else "CAUTION" if bool(caution[0]) else "task")
      self._log(
        f"[filter] step={self._i:6d} | V_safe[env0]={float(v_safe[0]):+.3f} "
        f"eps={self._eps:+.3f}/caut={self._caution:+.3f} -> {mode:7s} | "
        f"override_ema={self._override_ema:.2f} "
        f"| cmd=({float(self._teleop[0]):+.2f},{float(self._teleop[1]):+.2f},"
        f"{float(self._teleop[2]):+.2f})")
    self._i += 1
    self.last_mode = ("SAFETY" if bool(unsafe[0])
                      else "CAUTION" if bool(caution[0]) else "task")
    return action


# --- Runner / checkpoint helpers -------------------------------------------


def _build_runner(task_id: str, env, device: str):
  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  return runner_cls(env, asdict(agent_cfg), device=device)


def _load(runner, path: str, device: str, critic: bool) -> None:
  p = Path(path).expanduser().resolve()
  if not p.exists():
    raise FileNotFoundError(f"Checkpoint not found: {p}")
  try:
    runner.load(str(p), load_cfg={"actor": True, "critic": critic},
                strict=True, map_location=device)
  except RuntimeError as e:
    if "std_param" not in str(e):
      raise
    # mjlab 1.1.x -> current drift: old checkpoints store the action std under
    # "distribution.std_param"; current MLPModel expects "std". Remap and load
    # the actor state dict directly (see memory feedback_mjlab_api).
    sd = torch.load(str(p), map_location=device, weights_only=False)
    actor_sd = dict(sd["actor_state_dict"])
    if "distribution.std_param" in actor_sd:
      actor_sd["std"] = actor_sd.pop("distribution.std_param")
    runner.alg.actor.load_state_dict(actor_sd, strict=True)
    if critic:
      runner.alg.critic.load_state_dict(sd["critic_state_dict"], strict=True)
    print(f"[filter] loaded {p.name} via std_param compat remap")


# --- Main -------------------------------------------------------------------


def run(cfg: FilteredPlayConfig) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = build_env_cfg(cfg.num_envs, walk_phase_period=cfg.walk_phase_period)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  # Runtime-toggleable terminations: wrap the termination manager's compute so
  # the env skips auto-resets while disabled (T key). Internal telemetry
  # (terminated/time_outs) still reflects the real values.
  term_state = {"enabled": not cfg.no_terminations}
  tm = env.unwrapped.termination_manager
  _orig_tm_compute = tm.compute

  def _toggled_compute(*a, **kw):
    d = _orig_tm_compute(*a, **kw)
    if not term_state["enabled"]:
      return torch.zeros_like(d)
    return d

  tm.compute = _toggled_compute
  if cfg.no_terminations:
    print("[filter] episode terminations START DISABLED (press T to enable)")

  # Pin terrain difficulty if requested.
  terrain = env.unwrapped.scene.terrain
  if cfg.level is not None and getattr(terrain, "terrain_origins", None) is not None:
    terrain.terrain_levels[:] = int(cfg.level)
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types]
    print(f"[filter] pinned terrain level = {cfg.level}")

  # Task (walking) policy: actor only, reads the grafted "actor" group.
  walk_runner = _build_runner(WALK_TASK, env, device)
  _load(walk_runner, cfg.walk_checkpoint, device, critic=False)
  walk_actor = walk_runner.get_inference_policy(device=device)

  # Safety policy + critic: read "proprioception"/"critic" groups.
  safe_runner = _build_runner(SAFE_TASK, env, device)
  _load(safe_runner, cfg.safe_checkpoint, device, critic=True)
  safe_actor = safe_runner.get_inference_policy(device=device)
  safe_runner.alg.eval_mode()
  safe_critic = safe_runner.alg.critic

  # Teleop state + command injection.
  teleop = torch.zeros(3, device=device)

  def set_command(vec: torch.Tensor) -> None:
    cmd = env.unwrapped.command_manager.get_command("twist")
    cmd[:, :3] = vec.unsqueeze(0)

  VX_RANGE = (-1.0, 3.2)  # fast_walker2 training range
  VY_RANGE = (-1.0, 1.0)
  WZ_RANGE = (-1.0, 1.0)

  def on_key(key: int) -> None:
    from mjlab.viewer.native.keys import (
      KEY_0, KEY_I, KEY_J, KEY_K, KEY_L, KEY_O, KEY_T, KEY_U,
    )
    if key == KEY_T:
      term_state["enabled"] = not term_state["enabled"]
      print(f"[filter] episode terminations "
            f"{'ENABLED' if term_state['enabled'] else 'DISABLED (ENTER to reset manually)'}",
            flush=True)
      return
    if key == KEY_I:
      teleop[0] = min(teleop[0] + 0.25, VX_RANGE[1])
    elif key == KEY_K:
      teleop[0] = max(teleop[0] - 0.25, VX_RANGE[0])
    elif key == KEY_U:
      teleop[1] = min(teleop[1] + 0.25, VY_RANGE[1])
    elif key == KEY_O:
      teleop[1] = max(teleop[1] - 0.25, VY_RANGE[0])
    elif key == KEY_J:
      teleop[2] = min(teleop[2] + 0.25, WZ_RANGE[1])
    elif key == KEY_L:
      teleop[2] = max(teleop[2] - 0.25, WZ_RANGE[0])
    elif key == KEY_0:
      teleop.zero_()

  robot = env.unwrapped.scene["robot"]

  def get_speed() -> torch.Tensor:
    return torch.norm(robot.data.root_link_lin_vel_w, dim=1)

  def get_dones() -> torch.Tensor:
    return env.unwrapped.termination_manager.dones

  # Distance to the gap cluster (starts at x=2.5 from the patch origin —
  # SafetyFilterTerrainCfg.approach_length).
  def get_gap_dist() -> torch.Tensor:
    origin_x = env.unwrapped.scene.env_origins[:, 0]
    return 2.5 - (robot.data.root_link_pos_w[:, 0] - origin_x)

  policy = ValueFilteredPolicy(
    walk_actor=walk_actor,
    safe_actor=safe_actor,
    safe_critic=safe_critic,
    epsilon=cfg.epsilon,
    teleop_cmd=teleop,
    command_setter=set_command,
    speed_getter=get_speed,
    dones_getter=get_dones,
    caution=cfg.caution,
    hysteresis=cfg.hysteresis,
    rest_speed=cfg.rest_speed,
    gap_dist_getter=get_gap_dist,
    arm_distance=cfg.arm_distance,
    armed_epsilon=cfg.armed_epsilon,
    instant=cfg.instant,
    log_fn=lambda s: print(s, flush=True),
    log_every=cfg.log_every,
    disabled=cfg.no_filter,
  )

  print(
    f"[filter] epsilon={cfg.epsilon}  filter={'OFF (baseline)' if cfg.no_filter else 'ON'}\n"
    "[filter] teleop: I/K fwd/back  J/L turn  U/O strafe  0 stop  "
    "T toggle-terminations  ENTER force-reset\n"
    "[filter] drive toward the gap: filter should hand over to the safety "
    "policy (jump or brake); with --no-filter the walker falls in.")

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer

  if resolved == "native":
    viewer = NativeMujocoViewer(env, policy, key_callback=on_key)
  elif resolved == "viser":
    print("[filter][WARN] viser viewer has no keyboard teleop; the command "
          "stays at its current value (use --viewer native for teleop).")
    viewer = ViserPlayViewer(env, policy)
  else:
    raise RuntimeError(f"Unsupported viewer: {resolved}")

  viewer.run()
  env.close()


def main():
  run(tyro.cli(FilteredPlayConfig, prog=sys.argv[0]))


if __name__ == "__main__":
  main()
