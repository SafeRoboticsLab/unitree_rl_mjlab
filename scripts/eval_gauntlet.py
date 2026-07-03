"""Evaluate how far the crossing policy runs on the progressive gauntlet.

Builds the gauntlet env (gaps grow / platforms shrink), loads the latest
go2_crossing checkpoint, launches the robot into gap_0, and measures the max
forward distance (and gaps crossed) each env reaches before falling.

Run (mjlab env):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_gauntlet.py \
        --num-envs 32 --video
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.isaacs_go2.gauntlet_terrain import GAUNTLET_CFG  # noqa: E402

TASK = "Unitree-Go2-Gauntlet"
_TRACK_END = 22.0  # ~ last gap far-edge (20.6) + last platform; caps physics blow-ups


def latest_checkpoint(pattern="logs/rsl_rl/go2_crossing/*/model_*.pt"):
  cks = glob.glob(os.path.join(_REPO, pattern))
  if not cks:
    raise FileNotFoundError(f"No checkpoints under {pattern}")
  return max(cks, key=os.path.getmtime)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--max-steps", type=int, default=750)  # 15 s @ 50 Hz
  p.add_argument("--checkpoint", default=None, help="default: latest go2_crossing model")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--video", action="store_true")
  p.add_argument("--debug0", action="store_true", help="print env-0 per-step trajectory")
  p.add_argument("--start-x", type=float, default=-0.30,
                 help="spawn x relative to gap_0 near edge (more negative = further "
                      "back on the approach, so the full takeoff arc is visible)")
  args = p.parse_args()

  ckpt = args.checkpoint or latest_checkpoint()
  print(f"[eval] checkpoint: {ckpt}")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.events["reset_base"].params["start_x"] = args.start_x
  agent_cfg = load_rl_cfg(TASK)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner

  render_mode = "rgb_array" if args.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
  vid_dir = os.path.join(_REPO, "logs", "gauntlet_eval")
  if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    env = VideoRecorder(env, video_folder=vid_dir, step_trigger=lambda s: s == 0,
                        video_length=args.max_steps, disable_logger=True)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  run_env = runner.env  # the (safety-rewrapped) env the policy runs on

  robot = env.unwrapped.scene["robot"]
  n = args.num_envs
  dev = args.device

  run_env.reset()
  obs, _ = run_env.reset()  # 2nd reset: initial reset doesn't apply reset events
  # Measure forward distance RELATIVE TO EACH EPISODE'S SPAWN (origin-independent:
  # env_origins don't map to the gauntlet patch and are re-assigned per reset).
  # Spawn is at pose x=start_x, i.e. |start_x| m before gap_0's near edge, so
  # distance past gap_0 = (pos - spawn) + start_x.
  spawn_x = robot.data.root_link_pos_w[:, 0].clone()
  sx = args.start_x
  prev_raw = torch.full((n,), sx, device=dev)  # last plausible spawn-rel x
  max_dist = torch.zeros(n, device=dev)
  done_once = torch.zeros(n, dtype=torch.bool, device=dev)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  e0_raw = -1.0
  e0_fall_step = -1
  for t in range(args.max_steps):
    with torch.no_grad():
      act = policy(obs)
    obs, _rew, dones, _ex = run_env.step(act)
    dones = dones.bool()
    active = ~done_once  # snapshot before this step's freeze/reset updates
    pos = robot.data.root_link_pos_w
    vel = robot.data.root_link_lin_vel_w
    base_z = pos[:, 2].nan_to_num(-9.0)
    # distance past gap_0's near edge, relative to spawn (origin-independent)
    raw = (pos[:, 0] - spawn_x + sx).nan_to_num(-9.0)
    # Latch a permanent freeze on any physically-impossible jump (>50 m/s over a
    # 0.02 s step): a contact-solver blow-up teleports the base, so once an env
    # jumps we stop trusting it entirely (freeze its metric) rather than let a
    # late frame slip past the on-ground filter. A blow-up counts as a fall.
    blew = active & ((raw - prev_raw).abs() > 1.0)
    prev_raw = torch.where(blew, prev_raw, raw)
    x_rel = raw.clamp(-1.0, _TRACK_END)
    up = (-robot.data.projected_gravity_b[:, 2]).nan_to_num(-9.0)
    vz = vel[:, 2].nan_to_num(99.0)
    vx = vel[:, 0].nan_to_num(99.0)
    # Count progress only while genuinely upright + standing on a platform at a
    # plausible speed. A hard cap on distance-vs-elapsed-time is the decisive
    # guard: no robot can be past (start + 6 m/s * t), so any blow-up frame
    # (garbage position) is rejected regardless of what else it fakes.
    plaus_max = 2.0 + t * 0.02 * 6.0  # start slack + 6 m/s * elapsed seconds
    on_ground = ((up > 0.85) & (base_z > 0.18) & (base_z < 0.45)
                 & (vz.abs() < 1.5) & (vx > -0.5) & (vx < 4.5)
                 & (x_rel < plaus_max))
    good = active & on_ground & ~blew
    max_dist = torch.where(good, torch.maximum(max_dist, x_rel), max_dist)
    # env-0 ground truth (unclamped reach while stably upright; first fall step)
    if (not done_once[0]) and bool(on_ground[0]):
      e0_raw = max(e0_raw, float((pos[0, 0] - spawn_x[0] + sx).item()))
    if e0_fall_step < 0 and bool(dones[0]):
      e0_fall_step = t
    term = env.unwrapped.termination_manager.terminated.bool()
    if args.debug0 and not done_once[0]:
      if t % 4 == 0 or bool(dones[0]):
        print(f"  t={t:3d} x={float((pos[0,0]-spawn_x[0]+sx)):6.2f} "
              f"z={float(base_z[0]):5.2f} up={float(up[0]):5.2f} "
              f"vx={float(vx[0]):5.2f} vz={float(vz[0]):6.2f} "
              f"onG={bool(on_ground[0])} done={bool(dones[0])}")
      if bool(dones[0]):
        tm = env.unwrapped.termination_manager
        fired = [k for k in tm.active_terms if bool(tm.get_term(k)[0])]
        print(f"  >> env0 terminated at t={t}; terms fired: {fired or '(timeout/unknown)'}")
    newly = active & dones
    fell = torch.where(newly, term, fell)
    fell = torch.where(blew, torch.ones_like(fell), fell)  # blow-up = a fall
    done_once |= dones | blew
    if bool(done_once.all()):
      break
  print(f"[env0 truth] real reach (upright, unclamped): {e0_raw:.2f} m   "
        f"first-fall step: {e0_fall_step} (~{e0_fall_step*0.02:.1f}s)")

  md = max_dist.cpu().numpy()
  edges = np.array(GAUNTLET_CFG.gap_far_edges())
  gaps = np.array([int((md_i > edges).sum()) for md_i in md])
  fell_np = fell.cpu().numpy()

  print("\n=== GAUNTLET EVAL ===")
  print(f"envs: {n}   fell: {int(fell_np.sum())}/{n} (rest timed out standing)")
  print(f"max forward distance (m): mean {md.mean():.2f}  median {np.median(md):.2f}  "
        f"best {md.max():.2f}  worst {md.min():.2f}")
  print(f"gaps crossed:            mean {gaps.mean():.1f}  median {int(np.median(gaps))}  "
        f"best {int(gaps.max())}  worst {int(gaps.min())}")
  hist = np.bincount(gaps, minlength=GAUNTLET_CFG.n_gaps + 1)
  print("gaps-crossed histogram (#envs by gaps): "
        + "  ".join(f"{g}:{int(c)}" for g, c in enumerate(hist) if c > 0))
  print(f"[VIDEO env 0] gaps crossed: {int(gaps[0])}/{GAUNTLET_CFG.n_gaps}  "
        f"reach {md[0]:.2f} m  fell: {bool(fell_np[0])}")
  print(f"gap widths (m):  {[round(GAUNTLET_CFG.gap_start + i*GAUNTLET_CFG.gap_growth,2) for i in range(GAUNTLET_CFG.n_gaps)]}")
  print(f"gap far-edges (m): {[round(float(e),2) for e in edges]}")
  if args.video:
    print(f"video -> {vid_dir}")
  env.close()


if __name__ == "__main__":
  main()
