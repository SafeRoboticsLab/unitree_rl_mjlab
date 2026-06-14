"""Custom reset events for the Go2 + Piper payload task.

The default ``reset_joints_by_offset`` only writes qpos. For a passive
payload held in place by high-stiffness position actuators (piper arm),
the actuators read their setpoint from ``data.joint_pos_target`` every
step, so a randomized qpos gets snapped back to the default within a
handful of physics steps. This event writes both qpos **and**
``joint_pos_target`` so the arm stays where we placed it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_static_arm_pose(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset a group of joints to a random pose AND pin their actuator setpoint.

  Samples a per-env uniform offset in ``position_range`` around each joint's
  default pose, clamps to the soft joint limits, and writes the result to
  both ``qpos`` and ``joint_pos_target``. Velocity is zeroed.

  This is the right event for rigid-payload joints held by high-stiffness
  position actuators: the actuators will now hold the randomized pose
  instead of driving the arm back to its default.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids_tensor = torch.tensor(joint_ids, device=env.device)
  else:
    joint_ids_tensor = joint_ids  # slice

  joint_pos = default_joint_pos[env_ids][:, joint_ids_tensor].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  limits = soft_joint_pos_limits[env_ids][:, joint_ids_tensor]
  joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

  zeros = torch.zeros_like(joint_pos)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    zeros.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids_tensor,
  )
  # Pin the high-stiffness actuator setpoints so they hold the new pose
  # instead of pulling back to the default. set_joint_position_target
  # expects env_ids and joint_ids to be broadcast-compatible, so we index
  # the underlying tensor directly with the [N,1] x [M] pattern used by
  # EntityData._resolve_env_ids.
  asset.data.joint_pos_target[env_ids[:, None], joint_ids_tensor] = joint_pos
