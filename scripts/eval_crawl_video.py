"""Legible eval videos for the crawl safety task, one scenario per clip.

Scenarios (all in-band spawns via the reset hook; episodes auto-chain inside
each recording, so one clip shows several attempts):

  approach-slow   passable 0.35 m bar, arrive 0.8 m/s from 1.8 m
  approach-fast   passable 0.35 m bar, arrive 2.5 m/s
  settle-easy     crouched, fully cleared past the bar, 0.5 m/s (easy rung)
  under-beam      crouched under the beam, 1.0 m/s (crawl out forward)
  impossible      0.15 m bar (impossible row), arrive 1.5 m/s -> must STOP

Run (mjlab env):
    MUJOCO_GL=egl python scripts/eval_crawl_video.py                 # all
    MUJOCO_GL=egl python scripts/eval_crawl_video.py --scenario settle-easy
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
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.isaacs_go2.crawl_filter_terrain import BAR_DEPTH  # noqa: E402
from src.tasks.parkour.rl.reach_avoid_runner import (  # noqa: E402
  ParkourReachRestOnPolicyRunner,
)

TASK = "Unitree-Go2-Crawl"

# name -> (terrain row, _eval_spawn dict)
SCENARIOS = {
  "approach-slow": (0, {"d": 1.8, "v": 0.8}),
  "approach-fast": (0, {"d": 1.8, "v": 2.5}),
  "settle-easy": (0, {"d": -(BAR_DEPTH + 0.6), "v": 0.5, "crouch": True}),
  "under-beam": (2, {"d": -(BAR_DEPTH * 0.4), "v": 1.0, "crouch": True,
                     "crouch_alpha": 0.0}),
  "impossible": (9, {"d": 1.8, "v": 1.5}),
}


def latest_checkpoint():
  cks = glob.glob(os.path.join(_REPO, "logs/rsl_rl/go2_crawl/*/model_*.pt"))
  if not cks:
    raise FileNotFoundError("no go2_crawl checkpoints")
  return max(cks, key=os.path.getmtime)


def record(name, row, spawn, ckpt, args):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.curriculum = {}
  # Side-on follow camera: bar, colored zones, and body height all legible.
  env_cfg.viewer.distance = 3.6
  env_cfg.viewer.elevation = -18.0
  env_cfg.viewer.azimuth = 105.0
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  vid_dir = os.path.join(_REPO, "logs", "crawl_eval")
  os.makedirs(vid_dir, exist_ok=True)
  tag = os.path.basename(ckpt).replace(".pt", "")
  env = VideoRecorder(env, video_folder=vid_dir, step_trigger=lambda s: s == 0,
                      video_length=args.steps, disable_logger=True,
                      name_prefix=f"crawl-{name}-{tag}")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = ParkourReachRestOnPolicyRunner(env, asdict(agent_cfg), device=args.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=False, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  run_env = runner.env
  raw = env.unwrapped

  terrain = raw.scene.terrain
  terrain.terrain_levels[:] = row
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types]
  raw._eval_spawn = spawn
  run_env.reset()
  obs = run_env.get_observations().to(args.device)

  robot = raw.scene["robot"]
  tm = raw.termination_manager
  n_rest = n_fall = 0
  for t in range(args.steps):
    with torch.no_grad():
      act = policy(obs)
    obs, _g, dones, extras = run_env.step(act)
    if bool(dones.any()):
      t_o = extras.get("time_outs", torch.zeros_like(dones)).bool()
      rested = tm.get_term("rested_in_target").bool() if "rested_in_target" in tm.active_terms else t_o
      n_rest += int((dones.bool() & rested).sum())
      n_fall += int((dones.bool() & ~t_o).sum())
  env.close()
  print(f"[{name}] row {row} spawn {spawn} -> rested {n_rest}, fell/struck {n_fall} "
        f"(over {args.steps * 0.02:.0f} s x {args.num_envs} envs)")
  vids = sorted(glob.glob(os.path.join(vid_dir, f"crawl-{name}-{tag}*.mp4")),
                key=os.path.getmtime)
  return vids[-1] if vids else None


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--scenario", default=None, choices=list(SCENARIOS) + [None])
  p.add_argument("--checkpoint", default=None)
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--steps", type=int, default=500, help="frames (10 s)")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  ckpt = args.checkpoint or latest_checkpoint()
  print(f"[eval] checkpoint: {ckpt}")
  names = [args.scenario] if args.scenario else list(SCENARIOS)
  out = []
  for name in names:
    row, spawn = SCENARIOS[name]
    v = record(name, row, spawn, ckpt, args)
    if v:
      out.append(v)
  print("videos:")
  for v in out:
    print(" ", v)


if __name__ == "__main__":
  main()
