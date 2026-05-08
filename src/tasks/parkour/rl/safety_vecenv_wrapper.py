"""Safety vector environment wrapper for Go2 parkour.

Replaces the standard parkour reward with a safety margin g(s) encoding
actual physical failure for the Go2 quadruped traversing parkour terrain:

    g(s) = min(
        base_height_margin,      # base_link above min_base_height (fall into gap)
        orientation_margin,      # body tilt below tilt_limit
        contact_margin,          # no non-foot body touching obstacles/ground
    )

Safe iff g(s) >= 0.  The episode terminates when g(s) < 0.

**Terminal-state handling.** The underlying ManagerBasedRlEnv auto-resets
terminated environments within its step() call.  This means robot state
accessible AFTER the step is from the NEW episode, not the terminal
state.  For terminated (not timed-out) environments, we override g(s)
with a fixed negative value so the Safety Bellman Backup correctly
learns that termination = failure.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mjlab.entity import Entity
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor


class ParkourSafetyVecEnvWrapper(RslRlVecEnvWrapper):
  """Wraps a ManagerBasedRlEnv and replaces reward with g(s) for Go2 parkour.

  The safety margin combines three physical failure conditions:

      g(s) = min(base_height_margin, orientation_margin, contact_margin)

  ``base_height_margin``:
      ``base_link_z - min_base_height``.  Becomes negative when the
      robot falls into a gap or collapses.
  ``orientation_margin``:
      ``sin(tilt_limit) - |gravity_xy|``.  Becomes negative when the
      robot flips beyond the allowed tilt.
  ``contact_margin``:
      ``(threshold - max(nonfoot_contact_force)) / threshold``.  Becomes
      negative when a non-foot body part strikes the terrain/obstacles
      hard.

  Parameters
  ----------
  clip_actions:
      Forwarded to :class:`RslRlVecEnvWrapper`.
  min_base_height:
      Minimum base link height (m) before the margin goes negative.
      Should be lower than a normal standing height to give a positive
      signal on the plateau and drop sharply over gaps.
  tilt_limit_rad:
      Maximum allowable tilt from vertical (radians).
  terminal_margin:
      Fixed negative g(s) value assigned to terminated (not timed-out)
      environments.  Since the env auto-resets, we can't read the true
      terminal state — this value anchors the Bellman backup so that
      termination = negative safety value.
  nonfoot_contact_sensor_name:
      Name of the contact sensor that detects non-foot body contact.
      Set to ``None`` to disable the contact margin entirely.
  contact_force_threshold:
      Contact force (N) at which the contact margin crosses zero.
  """

  def __init__(
    self,
    env,
    *,
    clip_actions: float | None = None,
    min_base_height: float = 0.10,
    tilt_limit_rad: float = math.radians(70.0),
    terminal_margin: float = -0.1,
    nonfoot_contact_sensor_name: str | None = "nonfoot_ground_touch",
    contact_force_threshold: float = 10.0,
  ) -> None:
    super().__init__(env, clip_actions=clip_actions)

    self._min_base_height = float(min_base_height)
    self._sin_tilt_limit = math.sin(tilt_limit_rad)
    self._terminal_margin = float(terminal_margin)
    self._nonfoot_sensor_name = nonfoot_contact_sensor_name
    self._contact_force_threshold = float(contact_force_threshold)

    robot: Entity = self.unwrapped.scene["robot"]

    # Resolve base_link body index for the height margin.
    base_ids, _ = robot.find_bodies(("base_link",))
    if len(base_ids) == 0:
      self._base_body_id = 0
    else:
      first = base_ids[0]
      self._base_body_id = int(first.item()) if hasattr(first, "item") else int(first)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Step the environment and replace the reward with g(s)."""
    obs_dict, _reward, dones, extras = super().step(actions)

    extras["original_reward"] = _reward.clone()

    robot: Entity = self.unwrapped.scene["robot"]

    # --- Base height margin ---
    base_height = robot.data.body_link_pose_w[:, self._base_body_id, 2]
    base_height_margin = base_height - self._min_base_height  # (num_envs,)

    # --- Orientation margin ---
    proj_grav = robot.data.projected_gravity_b
    grav_xy_norm = torch.norm(proj_grav[:, :2], dim=1)
    orientation_margin = self._sin_tilt_limit - grav_xy_norm

    # --- Non-foot contact margin (optional) ---
    contact_margin: torch.Tensor | None = None
    if self._nonfoot_sensor_name is not None:
      try:
        sensor: ContactSensor = self.unwrapped.scene[self._nonfoot_sensor_name]
        data = sensor.data
        if data.force_history is not None:
          force_mag = torch.norm(data.force_history, dim=-1)
          # Reduce over any trailing dims (slots, history) → (num_envs,).
          while force_mag.dim() > 1:
            force_mag = force_mag.amax(dim=-1)
          peak_force = force_mag
        elif data.force is not None:
          force_mag = torch.norm(data.force, dim=-1)
          while force_mag.dim() > 1:
            force_mag = force_mag.amax(dim=-1)
          peak_force = force_mag
        else:
          peak_force = torch.zeros_like(base_height_margin)
        contact_margin = (
          self._contact_force_threshold - peak_force
        ) / self._contact_force_threshold
      except (KeyError, AttributeError):
        contact_margin = None

    # g(s) = weakest link.
    g = torch.minimum(base_height_margin, orientation_margin)
    if contact_margin is not None:
      g = torch.minimum(g, contact_margin)

    # --- Terminal-state correction ---
    # Environments terminated by the termination manager have already
    # been auto-reset; their post-step state reflects the NEW episode.
    # Override g(s) with the fixed terminal value so the Safety Bellman
    # Backup correctly associates termination with failure.
    time_outs = extras.get("time_outs", torch.zeros_like(dones))
    terminated_not_timeout = dones.bool() & ~time_outs.bool()
    g = torch.where(
      terminated_not_timeout,
      torch.full_like(g, self._terminal_margin),
      g,
    )

    # Log margin components for monitoring.
    log = extras.get("log", {})
    log["Safety/g"] = g.mean()
    log["Safety/base_height_margin"] = base_height_margin.mean()
    log["Safety/orientation_margin"] = orientation_margin.mean()
    if contact_margin is not None:
      log["Safety/contact_margin"] = contact_margin.mean()
    log["Safety/safe_rate"] = (g >= 0).float().mean()
    log["Safety/terminated_rate"] = terminated_not_timeout.float().mean()
    log["Safety/base_height"] = base_height.mean()
    extras["log"] = log

    return obs_dict, g, dones, extras
