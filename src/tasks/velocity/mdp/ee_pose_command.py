"""End-effector pose command for whole-body locomotion-manipulation control.

Mirrors the design used in ManipLoco (Fu et al., CoRL 2022):
  * Each env samples an ``end`` EE position in spherical coordinates
    (l, pitch, yaw) anchored at the arm-base body, expressed in a
    body-yaw-only frame at a fixed reference height.
  * ``Ttraj`` seconds later (sampled per resample), a new end is
    drawn; in between, the command is linearly interpolated from the
    previous end → current end so the policy sees a continuously
    sliding target rather than a step.
  * Orientation command is uniform on SO(3) (resampled together with
    position).

The ``command`` exposed to obs/reward is a 7-vector
``[p_cmd_world, o_cmd_world]`` in world frame so reward functions can
diff it against the gripper-site world pose directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_angle_axis,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _random_quat(n: int, device: torch.device) -> torch.Tensor:
  """Uniform random unit quaternion (w, x, y, z), shape (n, 4)."""
  u = torch.rand(n, 3, device=device)
  s1 = torch.sqrt(1.0 - u[:, 0])
  s2 = torch.sqrt(u[:, 0])
  q = torch.stack(
    [
      s2 * torch.cos(2 * math.pi * u[:, 2]),
      s1 * torch.sin(2 * math.pi * u[:, 1]),
      s1 * torch.cos(2 * math.pi * u[:, 1]),
      s2 * torch.sin(2 * math.pi * u[:, 2]),
    ],
    dim=-1,
  )
  return q


def _spherical_to_cartesian(
  l: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
  """(l, pitch, yaw) → (x, y, z). pitch around y, yaw around z, applied to +x."""
  cp = torch.cos(pitch)
  x = l * cp * torch.cos(yaw)
  y = l * cp * torch.sin(yaw)
  z = -l * torch.sin(pitch)
  return torch.stack([x, y, z], dim=-1)


class UniformEEPoseCommand(CommandTerm):
  """End-effector pose command anchored at an arm-base body."""

  cfg: "UniformEEPoseCommandCfg"

  def __init__(self, cfg: "UniformEEPoseCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # Resolve the anchor body and the EE site once.
    body_ids, _ = self.robot.find_bodies(cfg.anchor_body_name)
    assert len(body_ids) == 1, (
      f"anchor_body_name={cfg.anchor_body_name!r} did not match exactly one body."
    )
    self._anchor_body_id: int = body_ids[0]

    site_ids, _ = self.robot.find_sites(cfg.ee_site_name)
    assert len(site_ids) == 1, (
      f"ee_site_name={cfg.ee_site_name!r} did not match exactly one site."
    )
    self._ee_site_id: int = site_ids[0]

    # State tensors, all in world frame.
    self.p_start_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.p_end_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.q_end_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.q_end_w[:, 0] = 1.0  # identity
    self.p_cmd_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.q_cmd_w = self.q_end_w.clone()

    self.traj_total = torch.full(
      (self.num_envs,),
      0.5 * (cfg.resampling_time_range[0] + cfg.resampling_time_range[1]),
      device=self.device,
    )
    self.traj_elapsed = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_ee_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ee_quat"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """7-D world-frame target: [p_x, p_y, p_z, q_w, q_x, q_y, q_z]."""
    return torch.cat([self.p_cmd_w, self.q_cmd_w], dim=-1)

  # --- internal helpers ------------------------------------------------------

  def _anchor_frame(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position of the anchor body and yaw-only quaternion at those envs."""
    pos_w = self.robot.data.body_link_pos_w[env_ids, self._anchor_body_id]
    quat_w = self.robot.data.body_link_quat_w[env_ids, self._anchor_body_id]
    yaw_only = yaw_quat(quat_w)
    return pos_w, yaw_only

  def _sample_end_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fresh (p_end_w, q_end_w) for the given env ids."""
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    l = r.uniform_(*self.cfg.ranges.l).clone()
    pitch = r.uniform_(*self.cfg.ranges.pitch).clone()
    yaw = r.uniform_(*self.cfg.ranges.yaw).clone()
    p_local = _spherical_to_cartesian(l, pitch, yaw)

    anchor_pos_w, anchor_yaw_quat_w = self._anchor_frame(env_ids)
    p_end_w = anchor_pos_w + quat_apply(anchor_yaw_quat_w, p_local)
    if self.cfg.fixed_z is not None:
      p_end_w[:, 2] = self.cfg.fixed_z + p_local[:, 2]

    q_end_w = _random_quat(n, self.device)
    return p_end_w, q_end_w

  # --- CommandTerm overrides -------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    # Use the *current* command position as the new trajectory start so the
    # command stays continuous when a fresh end is drawn mid-flight.
    self.p_start_w[env_ids] = self.p_cmd_w[env_ids]
    p_end, q_end = self._sample_end_pose_w(env_ids)
    self.p_end_w[env_ids] = p_end
    self.q_end_w[env_ids] = q_end
    self.traj_total[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.resampling_time_range
    )
    self.traj_elapsed[env_ids] = 0.0

  def _update_command(self) -> None:
    self.traj_elapsed = self.traj_elapsed + self._env.step_dt
    alpha = (self.traj_elapsed / self.traj_total).clamp(0.0, 1.0).unsqueeze(-1)
    self.p_cmd_w = (1.0 - alpha) * self.p_start_w + alpha * self.p_end_w
    # Orientation: snap to the latest end (per-paper, only position is
    # interpolated). Could use slerp later if needed.
    self.q_cmd_w = self.q_end_w

  def _update_metrics(self) -> None:
    ee_pos_w = self.robot.data.site_pos_w[:, self._ee_site_id]
    ee_quat_w = self.robot.data.site_quat_w[:, self._ee_site_id]
    self.metrics["error_ee_pos"] += torch.norm(ee_pos_w - self.p_cmd_w, dim=-1)
    # Quaternion proximity = 1 - |<q1, q2>| (in [0, 1]).
    dot = torch.abs((ee_quat_w * self.q_cmd_w).sum(dim=-1)).clamp(max=1.0)
    self.metrics["error_ee_quat"] += 1.0 - dot

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    p_cmds = self.p_cmd_w.cpu().numpy()
    p_ees = self.robot.data.site_pos_w[:, self._ee_site_id].cpu().numpy()
    for batch in env_indices:
      visualizer.add_arrow(
        p_ees[batch], p_cmds[batch], color=(1.0, 0.6, 0.0, 0.8), width=0.01
      )


@dataclass(kw_only=True)
class UniformEEPoseCommandCfg(CommandTermCfg):
  """Config for :class:`UniformEEPoseCommand`."""

  entity_name: str
  """Scene entity to read poses from. Must contain ``anchor_body_name`` and
  ``ee_site_name``."""

  anchor_body_name: str = "piper_mount"
  """Name of the body whose position + yaw define the spherical-coord
  command frame."""

  ee_site_name: str = "gripper_site"
  """Site whose world pose is the actual EE pose, compared against the
  command pose for reward and metrics."""

  fixed_z: float | None = None
  """If set, override the world-z of the command frame to this value
  (paper uses 0.53). Useful so commands don't bob with the base."""

  @dataclass
  class Ranges:
    l: tuple[float, float] = (0.2, 0.7)
    pitch: tuple[float, float] = (-2 * math.pi / 5, 2 * math.pi / 5)
    yaw: tuple[float, float] = (-3 * math.pi / 5, 3 * math.pi / 5)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: "ManagerBasedRlEnv") -> UniformEEPoseCommand:
    return UniformEEPoseCommand(self, env)
