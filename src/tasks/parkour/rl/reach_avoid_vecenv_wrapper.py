"""Reach-avoid vector-env wrapper for Go2 parkour.

The avoid-only :class:`ParkourSafetyVecEnvWrapper` produces a single per-step
safety margin ``g(s)`` and, paired with the avoid-only Safety Bellman backup
``V(x) = min(g, V(x'))``, yields the *most conservative* policy: over a gap any
jump dips ``g`` mid-flight, so "never jump" strictly dominates and the policy
freezes at the edge.  To get a robot that crosses gaps / tracks a velocity
command *when it is safe to*, we need a **reach-avoid** signal.

This wrapper emits two margins per step:

    g(x)  (safety / avoid)  = min(
              base_height_margin,    # base above LOCAL ground (falls into gap)
              orientation_margin,    # body tilt below limit (tips over)
              contact_margin,        # no hard non-foot contact (wall/bar hit)
          )
    l(x)  (reach / target)  = liveness toward the velocity command:
              Phase 1 (progress):    forward speed >= a fraction of commanded vx
              Phase 2 (track, opt):  full xy velocity-command tracking

``g`` is returned as the reward (preserving the ``RslRlVecEnvWrapper``
interface that mjlab's runner expects); ``l`` is stashed in
``extras["target_margin"]`` so the paired :class:`ReachAvoidPPO` can apply the
backup ``V(x) = min( g(x), max( l(x), γ·V(x') ) )``.  ``V(x) > 0`` iff a policy
exists that drives the state into ``{l > 0}`` (moving with the command) while
never leaving ``{g > 0}`` (not falling / colliding / tipping).

**Terrain-relative safety margin.** Unlike the avoid-only wrapper (which used
absolute world-z and therefore conflated terrain elevation with safety on
parkour terrain), the base-height margin here is *relative to the local
platform* read from the ``terrain_scan`` raycast: the reference ground is the
highest surface within the robot footprint (the platform / gap edges, not the
distant gap floor).  A clean ballistic arc over a gap stays above the platform
edges → ``g > 0``; sinking into the gap drops the base below them → ``g < 0``.

References:
  J. F. Fisac et al., "Reach-Avoid Problems with Time-Varying Dynamics, Targets
  and Constraints", HSCC 2015.
  K.-C. Hsu et al., "Safety and Liveness Guarantees through Reach-Avoid
  Reinforcement Learning", RSS 2021.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mjlab.entity import Entity
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor


class ParkourReachAvoidVecEnvWrapper(RslRlVecEnvWrapper):
  """Wraps a parkour env, emitting terrain-relative ``g`` and command-reach ``l``.

  Parameters
  ----------
  clip_actions:
      Forwarded to :class:`RslRlVecEnvWrapper`.
  min_clearance:
      Minimum base height (m) above the *local* platform before the height
      margin goes negative.  Kept low enough to permit crawling under low
      barriers (the wall/bar *collision* hazard is handled by the contact
      margin, not the height margin).
  height_norm:
      Normalisation scale (m) for the height margin so it is comparable in
      magnitude to the (already unit-less) orientation/contact margins.
  tilt_limit_rad:
      Maximum allowable tilt from vertical (rad).
  terminal_margin:
      Fixed negative ``g`` assigned to terminated (not timed-out) envs.  The
      env auto-resets, so the true terminal state is unobservable; this anchors
      the Bellman backup so termination = failure.
  terrain_scan_sensor_name:
      Name of the down-looking raycast grid used to estimate local ground.
  footprint_radius:
      Planar radius (m) around the base over which raycast hits are considered
      "local ground".  Must exceed the half-width of the widest gap so a
      platform edge is always in view while crossing.
  nonfoot_contact_sensor_name:
      Contact sensor for non-foot body collisions (wall / bar / hard landing).
      ``None`` disables the contact margin.
  contact_force_threshold:
      Contact force (N) at which the contact margin crosses zero.
  command_name:
      Name of the velocity command term (twist: [vx, vy, wz] in body frame).
  reach_progress_fraction:
      Phase-1 reach target: the robot must move forward at >= this fraction of
      the commanded ``vx`` for ``l_progress >= 0``.
  reach_progress_floor:
      Minimum absolute forward speed (m/s) used as the progress target when the
      command is ~0, so standing still never counts as "at target".
  reach_progress_norm:
      Normalisation scale (m/s) for the progress margin.
  reach_track_tol:
      Phase-2 velocity-tracking tolerance (m/s): ``l_track = tol - |v_xy - cmd_xy|``.
  reach_track_weight:
      Curriculum knob in ``[0, 1]``.  ``0`` → reach is progress-only (Phase 1).
      ``>0`` → also require command tracking, ``l = min(l_progress, l_track)``
      with ``l_track`` slackened by ``(1 - weight)`` so it engages gradually.
  """

  def __init__(
    self,
    env,
    *,
    clip_actions: float | None = None,
    min_clearance: float = 0.08,
    height_norm: float = 0.25,
    tilt_limit_rad: float = math.radians(70.0),
    terminal_margin: float = -0.1,
    terrain_scan_sensor_name: str = "terrain_scan",
    footprint_radius: float = 0.5,
    obstacle_margin: float = 0.12,
    nonfoot_contact_sensor_name: str | None = "nonfoot_ground_touch",
    contact_force_threshold: float = 10.0,
    command_name: str = "twist",
    reach_progress_fraction: float = 0.5,
    reach_progress_floor: float = 0.2,
    reach_progress_norm: float = 0.5,
    reach_track_tol: float = 0.5,
    reach_track_weight: float = 0.0,
    reach_mode: str = "command",
    v_rest: float = 0.3,
    v_rest_norm: float = 0.5,
    cross_bias_weight: float = 0.3,
    cross_bias_scale: float = 3.0,
  ) -> None:
    super().__init__(env, clip_actions=clip_actions)

    self._min_clearance = float(min_clearance)
    self._height_norm = float(height_norm)
    self._sin_tilt_limit = math.sin(tilt_limit_rad)
    self._terminal_margin = float(terminal_margin)
    self._scan_name = terrain_scan_sensor_name
    self._footprint_radius = float(footprint_radius)
    self._obstacle_margin = float(obstacle_margin)
    self._nonfoot_sensor_name = nonfoot_contact_sensor_name
    self._contact_force_threshold = float(contact_force_threshold)
    self._command_name = command_name
    self._reach_progress_fraction = float(reach_progress_fraction)
    self._reach_progress_floor = float(reach_progress_floor)
    self._reach_progress_norm = float(reach_progress_norm)
    self._reach_track_tol = float(reach_track_tol)
    self._reach_track_weight = float(reach_track_weight)
    self._reach_mode = str(reach_mode)
    self._v_rest = float(v_rest)
    self._v_rest_norm = float(v_rest_norm)
    self._cross_bias_weight = float(cross_bias_weight)
    self._cross_bias_scale = float(cross_bias_scale)
    self._spawn_x: torch.Tensor | None = None  # per-env spawn x (rest mode)

    # Sanity-check the terrain scan sensor exists (terrain-relative g needs it).
    assert self._scan_name in self.unwrapped.scene.sensors, (
      f"Reach-avoid wrapper needs a raycast sensor named '{self._scan_name}'."
    )

  # --- Safety margin g(s) -------------------------------------------------

  def _base_height_margin(self, robot: Entity) -> torch.Tensor:
    """Base height above the LOCAL platform, from the terrain raycast.

    The reference ground is the *highest raycast hit that is still well below
    the base* within ``footprint_radius`` — i.e. the platform / gap edge the
    robot is traversing.  Surfaces within ``obstacle_margin`` of (or above) the
    base are excluded: those are walls / bars / step risers, not the floor, and
    must not inflate the reference (a wall beside the robot would otherwise read
    as zero clearance).  The far gap floor is excluded by taking the *highest*
    qualifying hit, so sinking into a gap (base drops below the platform edges)
    drives the margin negative while a clean arc over the gap keeps it positive.
    """
    scan = self.unwrapped.scene[self._scan_name]
    hit = scan.data.hit_pos_w  # (B, N, 3)
    dist = scan.data.distances  # (B, N); < 0 == miss

    base_z = robot.data.root_link_pos_w[:, 2]  # (B,)
    base_xy = robot.data.root_link_pos_w[:, None, :2]  # (B, 1, 2)
    planar = torch.norm(hit[..., :2] - base_xy, dim=-1)  # (B, N)
    in_footprint = (dist >= 0) & (planar <= self._footprint_radius)

    hit_z = hit[..., 2]
    # Platform candidates: footprint hits comfortably below the base.
    below = in_footprint & (hit_z < base_z[:, None] - self._obstacle_margin)
    neg_inf = torch.full_like(hit_z, -1.0e9)
    pos_inf = torch.full_like(hit_z, 1.0e9)
    ground_ref = torch.where(below, hit_z, neg_inf).max(dim=1).values  # (B,)

    # Fallback when nothing qualifies (e.g. base already collapsed onto the
    # platform): use the lowest footprint hit; if no hits at all, the base.
    lowest = torch.where(in_footprint, hit_z, pos_inf).min(dim=1).values
    lowest = torch.where(in_footprint.any(dim=1), lowest, base_z)
    ground_ref = torch.where(below.any(dim=1), ground_ref, lowest)

    clearance = base_z - ground_ref
    return clearance - self._min_clearance

  def _contact_margin(self, like: torch.Tensor) -> torch.Tensor | None:
    if self._nonfoot_sensor_name is None:
      return None
    try:
      sensor: ContactSensor = self.unwrapped.scene[self._nonfoot_sensor_name]
      data = sensor.data
      force = data.force_history if data.force_history is not None else data.force
      if force is None:
        peak = torch.zeros_like(like)
      else:
        mag = torch.norm(force, dim=-1)
        while mag.dim() > 1:
          mag = mag.amax(dim=-1)
        peak = mag
      return (self._contact_force_threshold - peak) / self._contact_force_threshold
    except (KeyError, AttributeError):
      return None

  def _safety_margin(self, robot: Entity) -> dict[str, torch.Tensor]:
    """Per-component (normalised) safety margins; ``g = min`` over them."""
    height = self._base_height_margin(robot) / self._height_norm

    proj_grav = robot.data.projected_gravity_b
    grav_xy = torch.norm(proj_grav[:, :2], dim=1)
    orientation = (self._sin_tilt_limit - grav_xy) / self._sin_tilt_limit

    margins = {"height": height, "orientation": orientation}
    contact = self._contact_margin(height)
    if contact is not None:
      margins["contact"] = contact
    return margins

  # --- Reach margin l(s) --------------------------------------------------

  def _target_margin(self, robot: Entity) -> dict[str, torch.Tensor]:
    """Reach target ``l``; ``l = min`` over the returned terms.

    ``reach_mode='rest'`` (safety-filter objective): the target is a SAFE STOP —
    ``l >= 0`` iff the robot is nearly at rest (``||v|| < v_rest``).  Combined
    with the reach-avoid backup ``V = min(g, max(l, γV'))``, "come to safe rest"
    is reached either by braking before a gap (if momentum allows) or by jumping
    to the far side and settling (if it does not — braking would drop ``g``
    below zero) or by chaining a cluster until solid ground is reached.  Jumping
    is thus *instrumental*, never rewarded directly.  A MILD additive bias toward
    resting further along (+x from spawn) breaks ties toward crossing when a
    crossing is safely available, without ever making motion itself a target
    (while moving, the rest term is strongly negative and the small bias cannot
    lift ``l`` above zero).

    ``reach_mode='command'`` (default): liveness toward the velocity command
    (forward-progress + optional tracking) — the original behavior.
    """
    if self._reach_mode == "rest":
      speed = torch.norm(robot.data.root_link_lin_vel_w, dim=1)  # full 3-D speed
      l_rest = (self._v_rest - speed) / self._v_rest_norm
      if self._spawn_x is None:
        self._spawn_x = robot.data.root_link_pos_w[:, 0].clone()
      prog = ((robot.data.root_link_pos_w[:, 0] - self._spawn_x)
              / self._cross_bias_scale).clamp(0.0, 1.0)
      return {"rest": l_rest + self._cross_bias_weight * prog}

    cmd = self.unwrapped.command_manager.get_command(self._command_name)  # (B, >=2)
    v_b = robot.data.root_link_lin_vel_b  # (B, 3)

    # Phase 1: forward progress at a fraction of the commanded speed.
    cmd_vx = cmd[:, 0]
    progress_target = torch.clamp(
      self._reach_progress_fraction * cmd_vx,
      min=self._reach_progress_floor,
    )
    progress = (v_b[:, 0] - progress_target) / self._reach_progress_norm
    margins = {"progress": progress}

    # Phase 2 (optional, curriculum): full xy command tracking.
    if self._reach_track_weight > 0.0:
      vel_err = torch.norm(v_b[:, :2] - cmd[:, :2], dim=1)
      # Slacken the tolerance by (1 - weight) so it engages gradually as the
      # curriculum raises ``reach_track_weight`` toward 1.
      tol = self._reach_track_tol * (2.0 - self._reach_track_weight)
      margins["track"] = (tol - vel_err) / self._reach_track_tol

    return margins

  # --- Step ---------------------------------------------------------------

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    obs_dict, reward, dones, extras = super().step(actions)
    extras["original_reward"] = reward.clone()

    robot: Entity = self.unwrapped.scene["robot"]

    safety = self._safety_margin(robot)
    target = self._target_margin(robot)
    g = torch.stack(list(safety.values()), dim=-1).min(dim=-1).values
    ell = torch.stack(list(target.values()), dim=-1).min(dim=-1).values

    # Terminal-state correction: env auto-resets, so override g for envs that
    # terminated (not timed out) with the fixed failure value.
    time_outs = extras.get("time_outs", torch.zeros_like(dones))
    terminated_not_timeout = dones.bool() & ~time_outs.bool()
    g = torch.where(
      terminated_not_timeout, torch.full_like(g, self._terminal_margin), g
    )

    # Rest mode: reset each env's spawn-x reference when it auto-resets, so the
    # cross-bias progress is measured from the new episode's spawn.
    if self._reach_mode == "rest" and self._spawn_x is not None:
      self._spawn_x = torch.where(
        dones.bool(), robot.data.root_link_pos_w[:, 0], self._spawn_x
      )

    # Reach-avoid algorithm reads l from here.
    extras["target_margin"] = ell

    log = extras.get("log", {})
    for k, v in safety.items():
      log[f"Safety/margin_{k}"] = v.mean()
    for k, v in target.items():
      log[f"Target/margin_{k}"] = v.mean()
    log["Safety/g"] = g.mean()
    log["Target/l"] = ell.mean()
    log["Safety/safe_rate"] = (g >= 0).float().mean()
    log["Target/at_goal_rate"] = (ell >= 0).float().mean()
    log["Safety/success_rate"] = ((g >= 0) & (ell >= 0)).float().mean()
    log["Safety/terminated_rate"] = terminated_not_timeout.float().mean()
    extras["log"] = log

    return obs_dict, g, dones, extras
