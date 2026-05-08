"""Safety-shield environment config.

Wraps :func:`unitree_go2_parkour_safety_env_cfg` with two targeted changes that
make the safety policy drop-in compatible with the walking policy at deploy
time.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.parkour.config.go2.safety_env_cfg import (
  unitree_go2_parkour_safety_env_cfg,
)


def unitree_go2_safety_shield_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_parkour_safety_env_cfg(play=play)

  # --- Restore height_scan in proprioception (shield deploy uses parkour sim
  #     with raycast, so we keep proprio structure identical to walking task). ---
  # Resolve the terrain_scan sensor to read its max_distance for the scale.
  scan_sensor = next(
    (
      s for s in (cfg.scene.sensors or ())
      if isinstance(s, RayCastSensorCfg) and s.name == "terrain_scan"
    ),
    None,
  )
  assert scan_sensor is not None, "terrain_scan sensor missing from safety env."

  cfg.observations["proprioception"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.03, n_max=0.03),
    scale=1 / scan_sensor.max_distance,
  )

  # --- Widen forward velocity to match walking-task distribution. ---
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.3, 1.2)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)

  return cfg
