"""Harvest mid-gait walking states from the fast walker = the TAKEOVER
distribution for the safety-policy handover finetune.

Rolls the frozen fast_walker2 policy in the gap world (level 0, flat approach)
with randomly resampled forward commands (0.5..3.2 m/s), and records healthy
mid-gait states: root height/orientation/velocities + joint pos/vel.  These are
replayed as a spawn stratum when finetuning the crossing-chain safety policy,
so the filter's mid-trot handovers are inside its training distribution.

Output: datasets/walker_handover_states.pt
  dict of tensors, each (K, ...): z, quat(wxyz), lin_vel_w(3), ang_vel_w(3),
  joint_pos(12), joint_vel(12), speed(1)
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import mjlab.tasks  # noqa: F401,E402
import src.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from play_filtered import (  # noqa: E402
  WALK_TASK, FilteredPlayConfig, _build_runner, _load, build_env_cfg,
)

TARGET = 150_000
N = 64
OUT = os.path.join(_REPO, "datasets", "walker_handover_states.pt")


def main():
  dev = "cuda:0"
  cfg = FilteredPlayConfig()
  env = ManagerBasedRlEnv(cfg=build_env_cfg(N, walk_phase_period=0.4), device=dev)
  env = RslRlVecEnvWrapper(env, clip_actions=None)
  terrain = env.unwrapped.scene.terrain
  terrain.terrain_levels[:] = 0
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types]

  runner = _build_runner(WALK_TASK, env, dev)
  _load(runner, cfg.walk_checkpoint, dev, critic=False)
  walk = runner.get_inference_policy(device=dev)
  robot = env.unwrapped.scene["robot"]

  obs, _ = env.reset()
  obs, _ = env.reset()
  cmd = env.unwrapped.command_manager.get_command("twist")
  vx = torch.empty(N, device=dev).uniform_(0.5, 3.2)

  buf = {k: [] for k in
         ("z", "quat", "lin_vel_w", "ang_vel_w", "joint_pos", "joint_vel", "speed")}
  kept = 0
  t = 0
  while kept < TARGET:
    if t % 100 == 0:  # resample commands (takeover diversity)
      vx = torch.empty(N, device=dev).uniform_(0.5, 3.2)
    cmd[:, 0] = vx
    cmd[:, 1] = 0.0
    cmd[:, 2] = 0.0
    with torch.no_grad():
      act = walk(obs)
    obs, _r, dones, _ex = env.step(act)

    if t % 3 == 0:  # decorrelate samples
      pos = robot.data.root_link_pos_w
      origin = env.unwrapped.scene.env_origins
      x_rel = pos[:, 0] - origin[:, 0]
      z = pos[:, 2]
      quat = robot.data.root_link_quat_w
      grav = robot.data.projected_gravity_b
      lin = robot.data.root_link_lin_vel_w
      ang = robot.data.root_link_ang_vel_w
      speed = torch.norm(lin, dim=1)
      # yaw within ~15 deg of +x (quat w,x,y,z: yaw ~ 2*atan2(qz, qw))
      yaw = 2.0 * torch.atan2(quat[:, 3], quat[:, 0])
      healthy = (
        (grav[:, 2] < -0.9)            # upright
        & (z > 0.22) & (z < 0.45)      # normal stance band, on ground
        & (speed > 0.3) & (speed < 3.4)
        & (x_rel > 0.3) & (x_rel < 2.2)  # on the approach, before the gap
        & (yaw.abs() < 0.26)
        & ~dones.bool()
      )
      if bool(healthy.any()):
        idx = healthy.nonzero().flatten()
        buf["z"].append(z[idx].cpu())
        buf["quat"].append(quat[idx].cpu())
        buf["lin_vel_w"].append(lin[idx].cpu())
        buf["ang_vel_w"].append(ang[idx].cpu())
        buf["joint_pos"].append(robot.data.joint_pos[idx].cpu())
        buf["joint_vel"].append(robot.data.joint_vel[idx].cpu())
        buf["speed"].append(speed[idx].cpu())
        kept += len(idx)
    if t % 600 == 0:
      print(f"t={t}  kept={kept}/{TARGET}", flush=True)
    t += 1

  data = {k: torch.cat(v)[:TARGET] for k, v in buf.items()}
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  torch.save(data, OUT)
  sp = data["speed"]
  print(f"saved {OUT}: {len(sp)} states, speed mean {sp.mean():.2f} "
        f"p10 {sp.quantile(0.1):.2f} p90 {sp.quantile(0.9):.2f}")
  env.close()


if __name__ == "__main__":
  main()
