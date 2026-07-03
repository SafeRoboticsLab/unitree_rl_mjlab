"""Reach-avoid wrapper for the gaps-only task, with a *foothold* reach margin.

Same safety margin ``g`` as :class:`ParkourReachAvoidVecEnvWrapper` (terrain-
relative height + tilt + non-foot contact), but the reach margin ``l`` is
redefined as **foothold support** rather than forward progress:

    l(x) = support_threshold - (base_z - ground_directly_under_robot)

``l >= 0`` iff there is solid ground *directly beneath* the robot (it is
standing on a platform), ``l < 0`` iff it is suspended over / fallen into a gap
(the nearest ground under the footprint center is far below).

Why this and not forward-progress: with a forward-progress reach term, launching
over a large gap momentarily satisfies the target (moving fast) while still
safe, so the reach-avoid value *rewards leaping* — the robot over-jumps.  With a
foothold target, "safely standing before an uncrossable gap" is itself a target
state (``l >= 0``), so it out-values falling in -> the policy learns to STOP at a
big gap, JUMP a small one (landing restores ``l >= 0``), and stop/continue once
across.  ``g`` (platform-edge reference) and ``l`` (under-robot reference) are
complementary: mid-clean-arc gives ``g >= 0`` (clearing) and ``l < 0`` (not yet
landed), exactly as intended.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity

from src.tasks.parkour.rl.reach_avoid_vecenv_wrapper import (
  ParkourReachAvoidVecEnvWrapper,
)


class GapReachAvoidVecEnvWrapper(ParkourReachAvoidVecEnvWrapper):
  """Reach-avoid wrapper whose reach margin is foothold support."""

  def __init__(
    self,
    env,
    *,
    central_radius: float = 0.20,
    support_threshold: float = 0.45,
    support_norm: float = 0.30,
    progress_weight: float = 0.5,
    patch_length: float = 8.0,
    **kwargs,
  ) -> None:
    super().__init__(env, **kwargs)
    self._central_radius = float(central_radius)
    self._support_threshold = float(support_threshold)
    self._support_norm = float(support_norm)
    self._progress_weight = float(progress_weight)
    self._patch_length = float(patch_length)

  def _target_margin(self, robot: Entity) -> dict[str, torch.Tensor]:
    scan = self.unwrapped.scene[self._scan_name]
    hit = scan.data.hit_pos_w  # (B, N, 3)
    dist = scan.data.distances  # (B, N)

    base_z = robot.data.root_link_pos_w[:, 2]
    base_xy = robot.data.root_link_pos_w[:, None, :2]
    planar = torch.norm(hit[..., :2] - base_xy, dim=-1)
    central = (dist >= 0) & (planar <= self._central_radius)

    hit_z = hit[..., 2]
    neg_inf = torch.full_like(hit_z, -1.0e9)
    # Closest (highest) supporting surface directly beneath the footprint.
    ground_under = torch.where(central, hit_z, neg_inf).max(dim=1).values
    # No ground under the centre (all miss / over a wide gap): treat as a deep
    # drop so the foothold margin is strongly negative.
    no_central = ~central.any(dim=1)
    ground_under = torch.where(no_central, base_z - 5.0, ground_under)

    support_distance = base_z - ground_under
    foothold = (self._support_threshold - support_distance) / self._support_norm

    # Forward progress along the patch.  Added to the foothold so that, AMONG
    # supported states, further-forward out-values nearer (the far platform
    # beats the near one -> cross a jumpable gap).  The foothold term is large
    # and negative when airborne, so this bonus never makes a mid-air launch a
    # target (launching a big gap stays < stopping at its edge).
    origin_x = self.unwrapped.scene.env_origins[:, 0]
    progress = ((robot.data.root_link_pos_w[:, 0] - origin_x) / self._patch_length).clamp(
      min=0.0, max=1.5
    )
    reach = foothold + self._progress_weight * progress
    # Single combined reach margin (parent aggregates target dict by min, so we
    # must not split foothold/progress into separate keys).
    return {"reach": reach}
