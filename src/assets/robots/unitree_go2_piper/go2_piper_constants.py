"""Unitree Go2 + Piper arm + onboard payload constants.

Combines the Go2 quadruped with the Piper 6-DOF arm mounted on the chassis
and a payload body representing battery + compute. The Piper joints are
articulated and held by high-stiffness position actuators; they are NOT
included in the RL action space. The walking policy continues to see and
control only the 12 Go2 joints.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_PIPER_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2_piper" / "xmls" / "go2_piper.xml"
)
assert GO2_PIPER_XML.exists()

GO2_JOINT_REGEX = (
  r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$"
)
"""Regex that matches only Go2 leg joints — the 12 joints the walking policy sees/controls."""

PIPER_JOINT_REGEX = r"^piper_joint[1-6]$"
"""Regex that matches only the 6 Piper arm joints (gripper fingers are welded)."""


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_PIPER_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_PIPER_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Go2 actuator config (matches go2_constants.py).
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*[LR]_hip_joint$",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*[LR]_thigh_joint$",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*[LR]_calf_joint$",),
  stiffness=200.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

##
# Piper arm actuator config. High stiffness so the arm holds its randomized
# initial pose throughout an episode (acts as a rigid payload). These
# actuators live on the robot entity but are excluded from the RL action
# space via the action term's actuator_names filter.
##

PIPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(PIPER_JOINT_REGEX,),
  stiffness=500.0,
  damping=30.0,
  effort_limit=200.0,
  armature=0.05,
)

# Whole-body variant: paper-spec gains so the piper joints are *actively
# controlled* by the policy rather than held rigid. Used by the
# Unitree-Go2-Piper-WholeBody-* tasks. Effort + armature are sized for the
# Piper's 6 DoF — see Piper datasheet; values mirror the WidowX 250s
# settings the ManipLoco paper uses (Kp=5, Kd=0.5).
PIPER_ACTUATOR_WHOLEBODY = BuiltinPositionActuatorCfg(
  target_names_expr=(PIPER_JOINT_REGEX,),
  stiffness=5.0,
  damping=0.5,
  effort_limit=50.0,
  armature=0.01,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    # Go2 legs — identical to go2_constants.py.
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
    # Piper default pose — a compact, stowed configuration so the arm does
    # not protrude wildly. Domain randomization perturbs these per-env.
    "piper_joint1": 0.0,
    "piper_joint2": 1.2,
    "piper_joint3": -1.4,
    "piper_joint4": 0.0,
    "piper_joint5": 0.6,
    "piper_joint6": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config — feet collide with ground (condim=3); other leg geoms
# collide with ground for illegal-contact termination. Piper and payload
# bodies do not collide (their geoms have contype=0/conaffinity=0 in the XML).
##

_foot_regex = "^[FR][LR]_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_PIPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
    PIPER_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO2_PIPER_WHOLEBODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
    PIPER_ACTUATOR_WHOLEBODY,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_piper_robot_cfg() -> EntityCfg:
  """Fresh Go2 + Piper + payload EntityCfg instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_PIPER_ARTICULATION,
  )


def get_go2_piper_wholebody_robot_cfg() -> EntityCfg:
  """Go2 + Piper EntityCfg for whole-body control: piper joints get low-Kp
  position actuators that the RL policy actively drives. The base XML and
  init state are shared with the rigid-payload variant; only the actuator
  parameters differ.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_PIPER_WHOLEBODY_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_piper_robot_cfg())
  viewer.launch(robot.spec.compile())
