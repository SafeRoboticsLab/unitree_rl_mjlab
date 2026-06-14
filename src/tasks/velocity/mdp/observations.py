from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_conjugate

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def ee_pose_b(
  env: ManagerBasedRlEnv,
  ee_site_name: str = "gripper_site",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Current end-effector pose in the base (root link) frame.

  Returns a (num_envs, 7) tensor [p_x, p_y, p_z, q_w, q_x, q_y, q_z].
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_ids, _ = asset.find_sites(ee_site_name)
  site_id = site_ids[0]
  ee_pos_w = asset.data.site_pos_w[:, site_id]
  ee_quat_w = asset.data.site_quat_w[:, site_id]
  base_pos_w = asset.data.root_link_pos_w
  base_quat_w = asset.data.root_link_quat_w
  ee_pos_b = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
  ee_quat_b = quat_mul(quat_conjugate(base_quat_w), ee_quat_w)
  return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def ee_command_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """End-effector pose *command* expressed in the base frame.

  The command term itself stores the target in world frame so reward
  functions can use site-world poses directly. The policy, however, is
  invariant to absolute world placement, so we rotate the target into the
  base frame for observation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None and command.shape[-1] == 7
  base_pos_w = asset.data.root_link_pos_w
  base_quat_w = asset.data.root_link_quat_w
  cmd_pos_b = quat_apply_inverse(base_quat_w, command[:, 0:3] - base_pos_w)
  cmd_quat_b = quat_mul(quat_conjugate(base_quat_w), command[:, 3:7])
  return torch.cat([cmd_pos_b, cmd_quat_b], dim=-1)


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

