"""Legible eval videos for the safety-filter rest objective.

Records one full episode of env 0 with a CONTROLLED arrival momentum, so the
learned decision is visible: slow arrival -> brake and stand; fast arrival ->
jump/chain to the rest zone and settle.

Run (mjlab env):
    MUJOCO_GL=egl python scripts/eval_rest_video.py --vx 0.5 --tag slow
    MUJOCO_GL=egl python scripts/eval_rest_video.py --vx 2.6 --tag fast
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import src.tasks  # noqa: E402,F401
import src.tasks.go2_safety_filter.crossing_chain.env_cfg as chain_cfg  # noqa: E402

TASK = "Unitree-Go2-Crossing-Chain"


def latest_checkpoint():
  cks = glob.glob(os.path.join(_REPO, "logs/rsl_rl/go2_crossing_chain/*/model_*.pt"))
  if not cks:
    raise FileNotFoundError("no go2_crossing_chain checkpoints")
  return max(cks, key=os.path.getmtime)  # newest across all runs


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--vx", type=float, default=2.6, help="arrival momentum (m/s)")
  p.add_argument("--tag", default="clip")
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--steps", type=int, default=420, help="frames (~8.4 s = 1 episode)")
  p.add_argument("--checkpoint", default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--level", type=int, default=None,
                 help="pin all envs to this terrain difficulty row (0=easiest)")
  p.add_argument("--spawn-x", type=float, default=None,
                 help="pin spawn x on the approach (default: random 0.2-1.8)")
  p.add_argument("--spawn-z", type=float, default=0.05, help="spawn height above ground")
  p.add_argument("--vz", type=float, default=0.0, help="initial vertical velocity")
  args = p.parse_args()


  ckpt = args.checkpoint or latest_checkpoint()
  print(f"[eval] checkpoint: {ckpt}   arrival vx ~ {args.vx}")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs

  # Fully controlled ground spawn: exact position + forward momentum, vz ~ 0.
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

  spawn_x_pin = args.spawn_x if args.spawn_x is not None else 1.0

  def pinned_reset(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    asset = env.scene[asset_cfg.name]
    device = env.device
    n = int(len(env_ids))
    root = asset.data.default_root_state[env_ids].clone()

    def u(lo, hi):
      return sample_uniform(lo, hi, (n,), device)

    pose = torch.stack(
      [spawn_x_pin + u(-0.02, 0.02), u(-0.04, 0.04), args.spawn_z + u(-0.01, 0.01),
       u(-0.03, 0.03), u(-0.03, 0.03), u(-0.04, 0.04)], dim=1)
    vel = torch.stack(
      [args.vx + u(-0.05, 0.05), u(-0.05, 0.05), args.vz + u(-0.03, 0.03),
       u(-0.05, 0.05), u(-0.05, 0.05), u(-0.05, 0.05)], dim=1)
    positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = quat_mul(
      root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root[:, 7:13] + vel, env_ids=env_ids)

  env_cfg.events["reset_base"] = EventTermCfg(func=pinned_reset, mode="reset", params={})
  # Legible follow camera: farther back, higher, side-on so braking vs jumping
  # and the colored terrain zones are visible.
  env_cfg.viewer.distance = 4.5
  env_cfg.viewer.elevation = -20.0
  env_cfg.viewer.azimuth = 90.0
  agent_cfg = load_rl_cfg(TASK)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  vid_dir = os.path.join(_REPO, "logs", "rest_eval")
  os.makedirs(vid_dir, exist_ok=True)
  env = VideoRecorder(env, video_folder=vid_dir, step_trigger=lambda s: s == 0,
                      video_length=args.steps, disable_logger=True,
                      name_prefix=f"rest-{args.tag}")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=False, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  run_env = runner.env

  robot = env.unwrapped.scene["robot"]
  terrain = env.unwrapped.scene.terrain
  if args.level is not None and getattr(terrain, "terrain_origins", None) is not None:
    terrain.terrain_levels[:] = int(args.level)
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    print(f"[eval] pinned terrain level = {args.level}")
  run_env.reset()
  obs, _ = run_env.reset()  # 2nd reset applies reset events (pinned vx)

  # Per-episode tracking for env 0 (episodes auto-chain inside the recording).
  ep_start_t = 0
  ep_start_x = robot.data.root_link_pos_w[0, 0].item()
  print(f"[env0] ep1 spawn vx = {robot.data.root_link_lin_vel_w[0, 0].item():.2f}")
  episodes = []
  for t in range(args.steps):
    with torch.no_grad():
      act = policy(obs)
    obs, _r, dones, _ex = run_env.step(act)
    if bool(dones[0]):
      term = bool(env.unwrapped.termination_manager.terminated[0])
      # position BEFORE the auto-reset is gone; use last step's value
      episodes.append({
        "start": ep_start_t, "end": t,
        "outcome": "FELL" if term else "SAFE REST",
        "disp": last_disp,
      })
      print(f"[env0] episode {len(episodes)}: steps {ep_start_t}-{t} "
            f"({ep_start_t*0.02:.1f}-{t*0.02:.1f}s)  {episodes[-1]['outcome']}  "
            f"forward {last_disp:.2f} m")
      ep_start_t = t + 1
      ep_start_x = robot.data.root_link_pos_w[0, 0].item()
    else:
      last_disp = robot.data.root_link_pos_w[0, 0].item() - ep_start_x
  env.close()
  print(f"video -> {vid_dir}")
  crossed = [e for e in episodes if e["outcome"] == "SAFE REST" and e["disp"] > 2.0]
  if crossed:
    e = crossed[0]
    print(f"[CUT] ffmpeg -ss {e['start']*0.02:.1f} -to {min(e['end']*0.02+0.5, args.steps*0.02):.1f} (crossed episode)")


if __name__ == "__main__":
  main()
