"""Parkour-specific termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate when non-foot body parts contact obstacles with force above threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  force_mag = torch.norm(data.force, dim=-1) if data.force is not None else data.found.float()
  return (force_mag > force_threshold).any(dim=-1)


def base_too_low(
  env: ManagerBasedRlEnv,
  min_height: float = 0.12,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when robot base drops below minimum height (fell into gap)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < min_height


def base_too_high(
  env: ManagerBasedRlEnv,
  max_height: float = 1.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when robot base is too high (launched/glitched)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] > max_height
