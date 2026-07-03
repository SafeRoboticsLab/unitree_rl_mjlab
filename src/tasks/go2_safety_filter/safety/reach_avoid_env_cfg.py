"""Reach-avoid environment config for the Go2 parkour safety policy.

Built on ``unitree_go2_parkour_safety_env_cfg`` (NOT the shield env): that config
keeps the actor observation **hardware-deployable** — proprioception (no raycast
``height_scan``) + the front depth camera (RGBD).  The raycast heightfield stays
in the *critic* group only (privileged, sim-only).  The reach-avoid safety/reach
margins ``g``/``l`` are computed in the wrapper from sim state (raycast + base
velocity); those are the training *reward signal*, not policy observations, so
using privileged data there does not affect deployability.

Reach-avoid-specific changes layered on top:

1. **Forward velocity command** so the reach term ``l`` has a meaningful target.
2. **Terrain-relative failure termination** (``base_below_local_terrain``) so a
   gap/cliff fall is detected regardless of platform elevation, matching
   ``g_height < 0``.
3. **Jump-trajectory boundary sampling** — mid-air-over-gap spawns plus forward
   velocity on the on-ground spawns, spread across approach/flight/landing.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.parkour.mdp as mdp
from src.tasks.parkour.config.go2.safety_env_cfg import (
  unitree_go2_parkour_safety_env_cfg,
)


def unitree_go2_reach_avoid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_parkour_safety_env_cfg(play=play)

  # --- Strict raw proprioception for the actor. ---
  # Drop absolute ``base_height`` (world-z) — not a raw onboard sensor.  It
  # stays in the privileged ``critic`` group (which has its own copy).  The
  # actor therefore sees only IMU + joints + command + phase + last action,
  # plus the depth image.
  cfg.observations["proprioception"].terms.pop("base_height", None)

  # --- Forward velocity command (the reach target tracks this). ---
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.3, 1.2)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)

  # --- Terrain-relative failure termination (matches g_height < 0). ---
  cfg.terminations.pop("base_too_low", None)
  cfg.terminations["base_below_local_terrain"] = TerminationTermCfg(
    func=mdp.base_below_local_terrain,
    params={
      "min_clearance": 0.08,
      "sensor_name": "terrain_scan",
      "footprint_radius": 0.5,
      "obstacle_margin": 0.12,
    },
  )

  # --- Jump-trajectory boundary sampling. ---
  # The parent already installs ``reset_robot_midair_over_gaps``.  Give the
  # on-ground spawns a forward velocity (so episodes begin *approaching* a gap,
  # committed but still safe) and widen the mid-air arc so both recoverable and
  # doomed flight states are seen across the patch.
  reset = cfg.events.get("reset_base")
  if reset is not None and not play:
    reset.params["ground_velocity_range"] = {"x": (0.3, 1.2), "y": (-0.2, 0.2)}
    reset.params["midair_fraction"] = 0.5
    reset.params["midair_x_range"] = (0.8, 7.0)
    reset.params["midair_z_range"] = (0.30, 0.90)
    reset.params["midair_vx_range"] = (0.8, 3.5)
    reset.params["midair_vz_range"] = (-1.5, 0.5)
    reset.params["midair_pitch_range"] = (-math.radians(20.0), math.radians(20.0))

  return cfg
