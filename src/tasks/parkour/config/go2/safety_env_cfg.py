"""Unitree Go2 parkour **safety** environment configuration.

Builds on the standard parkour environment and adapts it for safety
training with :class:`ParkourSafetyVecEnvWrapper` + :class:`SafetyPPO`:

* **Actor uses depth + proprioception only** — the raycast height scan
  is removed from the ``proprioception`` group (which feeds the policy)
  and kept only in the ``critic`` group as privileged information.
  The trained policy therefore depends exclusively on depth camera +
  proprioception and can be deployed without the raycast sensor.
* **Safety-friendly initialisation** — resets use a strategic mix of
  on-ground spawns and mid-air spawns above the terrain with forward
  velocity, so a sizeable fraction of episodes begin over a gap.
* **Stronger, more frequent pushes** — to expose the policy to
  meaningful perturbations that would push a walking task policy off a
  platform.
* **Low forward velocity command** — the training reward is the
  safety margin g(s); velocity tracking is not the objective, so the
  command range is tightened.

Shaped rewards defined in the base cfg are harmless: the safety
wrapper replaces the reward with g(s) regardless of their values.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import src.tasks.parkour.mdp as mdp
from src.tasks.parkour.config.go2.env_cfgs import unitree_go2_parkour_env_cfg


def _remove_from_group(cfg: ManagerBasedRlEnvCfg, group_name: str, term_name: str) -> None:
  """Drop ``term_name`` from the ``group_name`` observation group if present."""
  group = cfg.observations.get(group_name)
  if group is None or group.terms is None:
    return
  group.terms.pop(term_name, None)


def unitree_go2_parkour_safety_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 parkour safety configuration."""
  cfg = unitree_go2_parkour_env_cfg(play=play)

  # --- Actor policy must not see the raycast scan. ---
  # The raycast sensor remains in the scene; only the critic group is
  # allowed to consume it as privileged information.  At deployment
  # only the actor (depth + proprioception) is queried, so the policy
  # is raycast-independent.
  _remove_from_group(cfg, "proprioception", "height_scan")

  # Drop gap-crossing rewards that depend on the raycast sensor.
  # They are replaced by g(s) anyway, so removing them saves compute.
  cfg.rewards.pop("gap_crossing", None)
  cfg.rewards.pop("gap_crossing_bonus", None)

  # --- Safety-friendly termination ---
  # Keep `fell_over` (pitch/roll limit), `base_too_low` (fell into a
  # gap), and the existing `illegal_contact` termination — all three
  # are consistent with the safety wrapper's g(s).
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.base_too_low,
    params={"min_height": 0.10},
  )

  # --- Velocity command (low forward) ---
  # SafetyPPO learns from g(s), not from velocity tracking.  A small
  # positive vx command keeps the phase/gait observations meaningful
  # and lets the base walking task policy (if any) do its thing
  # underneath the safety filter.
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.0, 0.5)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)

  # --- Strategic initialisation (the crux of gap learning) ---
  # Replace the plain reset_base with the mid-air-over-gaps reset so
  # the policy is explicitly exposed to mid-flight states above gaps.
  # Without this, an on-ground spawn + slow forward command would
  # rarely place the robot in a true mid-jump state, and the policy
  # would never learn to bridge gaps under recovery.
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_robot_midair_over_gaps,
    mode="reset",
    params={
      "midair_fraction": 0.5,
      "ground_pose_range": {
        "x": (-0.3, 0.3),
        "y": (-0.2, 0.2),
        "z": (0.0, 0.0),
        "yaw": (-0.2, 0.2),
      },
      "ground_velocity_range": {
        "x": (-0.2, 0.5),
      },
      # Forward offsets span the 8 m patch — some spawns land over gaps.
      "midair_x_range": (1.0, 6.5),
      "midair_y_range": (-0.2, 0.2),
      # Go2 default standing z = 0.32 m; add 0.1-0.5 m for mid-flight.
      "midair_z_range": (0.40, 0.85),
      # Strong forward velocity so the robot is committed to crossing.
      "midair_vx_range": (1.5, 3.5),
      "midair_vy_range": (-0.2, 0.2),
      "midair_vz_range": (-1.0, 0.5),
      "midair_roll_range": (-0.2, 0.2),
      "midair_pitch_range": (-0.25, 0.25),
      "midair_yaw_range": (-0.3, 0.3),
    },
  )

  # Slightly more aggressive joint reset so the policy sees non-trivial
  # initial joint configurations in-flight.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.15, 0.15)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)

  # --- Stronger, more frequent pushes (only in training mode). ---
  if not play and "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (1.0, 3.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.8, 1.2),
      "y": (-0.8, 0.8),
      "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.5, 0.5),
    }

  if play:
    cfg.events.pop("push_robot", None)
    # In play mode, keep mid-air resets optional so the viewer starts
    # from the floor most of the time.
    cfg.events["reset_base"].params["midair_fraction"] = 0.25

  return cfg
