"""Evaluate an archived ISAACS controller against an archived worst-case adversary.

In ISAACS the leaderboard archive *is* the deliverable: the live policy can keep
oscillating, but the board retains the best controllers (by reach-avoid metric).
This loads an archived ctrl (and optionally a dstb) state-dict into a fresh
IsaacsSAC policy and reports reach-avoid robustness over N episodes:
  * nominal (no disturbance)
  * adversarial (vs the loaded worst-case dstb)

Run (mjlab env)::

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_isaacs_leaderboard.py \
        --run runs/0qkjpmii --ctrl 225024 --dstb 405056 --episodes 10 --video
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")
sys.path.insert(0, _REPO)

import imageio.v2 as imageio  # noqa: E402

from safety_sb3 import IsaacsSAC  # noqa: E402
from src.isaacs_go2 import Go2IsaacsEnv  # noqa: E402


def run_episodes(model, env, n, max_steps, use_dstb, record):
  safe, reached, succ, eplens, frames = [], [], [], [], []
  for ep in range(n):
    obs, info = env.reset()
    ep_safe, ep_reached, ln = True, False, 0
    for _ in range(max_steps):
      obs_t = torch.as_tensor(obs[None], device=model.device, dtype=torch.float32)
      with torch.no_grad():
        a_ctrl = model.policy.actor(obs_t, deterministic=True).cpu().numpy()[0]
        if use_dstb:
          a_dstb = model.policy.dstb_actor(obs_t, deterministic=True).cpu().numpy()[0]
        else:
          a_dstb = np.zeros(model.policy.dstb_action_dim, dtype=np.float32)
      obs, g, term, trunc, info = env.step(np.concatenate([a_ctrl, a_dstb]))
      ln += 1
      if g < 0.0:
        ep_safe = False
      if info["l_x"] >= 0.0:
        ep_reached = True
      if record and ep == 0:
        frames.append(np.asarray(env.render()))
      if term or trunc:
        break
    safe.append(float(ep_safe))
    reached.append(float(ep_reached))
    succ.append(float(ep_safe and ep_reached))
    eplens.append(ln)
  return {
    "safe_rate": float(np.mean(safe)),
    "reach_rate": float(np.mean(reached)),
    "success_rate": float(np.mean(succ)),
    "ep_len_mean": float(np.mean(eplens)),
  }, frames


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, help="run dir, e.g. runs/0qkjpmii")
  p.add_argument("--ctrl", type=int, required=True, help="archived ctrl step")
  p.add_argument("--dstb", type=int, default=None, help="archived dstb step (worst-case)")
  p.add_argument("--episodes", type=int, default=10)
  p.add_argument("--max-steps", type=int, default=400)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--ctrl-gain", type=float, default=3.0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--video", action="store_true")
  args = p.parse_args()

  env = Go2IsaacsEnv(
    render_mode="rgb_array" if args.video else None,
    device=args.device,
    force_max=args.force_max,
    ctrl_gain=args.ctrl_gain,
  )
  model = IsaacsSAC(
    "MlpPolicy",
    env,
    ctrl_action_dim=Go2IsaacsEnv.CTRL_DIM,
    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 512], qf=[256, 256, 256])),
    device=args.device,
    verbose=0,
  )

  lb = os.path.join(args.run, "leaderboard")
  ctrl_sd = torch.load(os.path.join(lb, f"ctrl_{args.ctrl}.pt"), map_location=args.device)
  model.policy.actor.load_state_dict(ctrl_sd)
  print(f"loaded ctrl_{args.ctrl}")
  if args.dstb is not None:
    dstb_sd = torch.load(os.path.join(lb, f"dstb_{args.dstb}.pt"), map_location=args.device)
    model.policy.dstb_actor.load_state_dict(dstb_sd)
    print(f"loaded dstb_{args.dstb}")

  nom, nom_frames = run_episodes(
    model, env, args.episodes, args.max_steps, use_dstb=False, record=args.video
  )
  print(f"NOMINAL    ({args.episodes} eps): {nom}")
  if args.dstb is not None:
    adv, adv_frames = run_episodes(
      model, env, args.episodes, args.max_steps, use_dstb=True, record=args.video
    )
    print(f"ADVERSARIAL({args.episodes} eps): {adv}")
  else:
    adv_frames = []

  if args.video:
    out = os.path.join(args.run, "eval")
    os.makedirs(out, exist_ok=True)
    if nom_frames:
      imageio.mimsave(os.path.join(out, f"ctrl{args.ctrl}_nominal.mp4"), nom_frames,
                      fps=50, macro_block_size=1)
    if adv_frames:
      imageio.mimsave(os.path.join(out, f"ctrl{args.ctrl}_vs_dstb{args.dstb}.mp4"),
                      adv_frames, fps=50, macro_block_size=1)
    print(f"videos -> {out}")
  env.close()


if __name__ == "__main__":
  main()
