"""Reconstruct the ISAACS leaderboard's reach-avoid score matrix at termination.

The live board (ctrl x dstb reach-avoid success) lives in memory and is lost when
the run is killed, but the archived ctrl/dstb actor state-dicts are on disk. This
re-scores every (ctrl_i, dstb_j) pair — plus the dummy (no-disturbance) column —
over ``num_envs`` PARALLEL episodes via Go2IsaacsVecEnv (much cleaner than the
3-episode live metric), and reports the matrix + the maximin (worst-case-robust)
controller selection.

Run (mjlab env)::

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_isaacs_matrix.py \
        --run runs/0qkjpmii --num-envs 64 --max-steps 400
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")
sys.path.insert(0, _REPO)

from safety_sb3 import IsaacsSAC  # noqa: E402
from src.isaacs_go2 import Go2IsaacsEnv, Go2IsaacsVecEnv  # noqa: E402


def _steps(lb_dir, kind):
  out = []
  for p in glob.glob(os.path.join(lb_dir, f"{kind}_*.pt")):
    m = re.search(rf"{kind}_(\d+)\.pt$", p)
    if m:
      out.append(int(m.group(1)))
  return sorted(out)


def eval_pair(model, env, ctrl_sd, dstb_sd, max_steps):
  """Reach-avoid success rate over env.num_envs parallel first-episodes."""
  model.policy.actor.load_state_dict(ctrl_sd)
  use_dstb = dstb_sd is not None
  if use_dstb:
    model.policy.dstb_actor.load_state_dict(dstb_sd)
  dn = model.policy.dstb_action_dim
  N = env.num_envs

  obs = env.reset()
  ep_safe = np.ones(N, dtype=bool)
  ep_reached = np.zeros(N, dtype=bool)
  done_once = np.zeros(N, dtype=bool)
  for _ in range(max_steps):
    obs_t = torch.as_tensor(obs, device=model.device, dtype=torch.float32)
    with torch.no_grad():
      c = model.policy.actor(obs_t, deterministic=True).cpu().numpy()
      d = (
        model.policy.dstb_actor(obs_t, deterministic=True).cpu().numpy()
        if use_dstb
        else np.zeros((N, dn), dtype=np.float32)
      )
    env.step_async(np.concatenate([c, d], axis=1).astype(np.float32))
    obs, g, dones, infos = env.step_wait()
    active = ~done_once
    ep_safe &= ~(active & (g < 0.0))
    lx = np.array([info["l_x"] for info in infos], dtype=np.float32)
    ep_reached |= active & (lx >= 0.0)
    done_once |= active & dones.astype(bool)
    if done_once.all():
      break
  return float(np.mean(ep_safe & ep_reached)), float(np.mean(ep_safe))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--max-steps", type=int, default=400)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  lb_dir = os.path.join(args.run, "leaderboard")
  ctrls = _steps(lb_dir, "ctrl")
  dstbs = _steps(lb_dir, "dstb")
  print(f"ctrl archive: {ctrls}")
  print(f"dstb archive: {dstbs}")

  env = Go2IsaacsVecEnv(num_envs=args.num_envs, device=args.device, force_max=args.force_max)
  model = IsaacsSAC(
    "MlpPolicy", Go2IsaacsEnv(device=args.device, force_max=args.force_max),
    ctrl_action_dim=Go2IsaacsEnv.CTRL_DIM,
    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 512], qf=[256, 256, 256])),
    device=args.device, verbose=0,
  )
  ctrl_sds = {s: torch.load(os.path.join(lb_dir, f"ctrl_{s}.pt"), map_location=args.device) for s in ctrls}
  dstb_sds = {s: torch.load(os.path.join(lb_dir, f"dstb_{s}.pt"), map_location=args.device) for s in dstbs}

  cols = dstbs + ["dummy"]
  succ = np.zeros((len(ctrls), len(cols)))
  safe = np.zeros((len(ctrls), len(cols)))
  for i, cs in enumerate(ctrls):
    for j, ds in enumerate(cols):
      dstb_sd = None if ds == "dummy" else dstb_sds[ds]
      succ[i, j], safe[i, j] = eval_pair(model, env, ctrl_sds[cs], dstb_sd, args.max_steps)

  def show(mat, title):
    print(f"\n=== {title} (rows=ctrl, cols=dstb; {args.num_envs} parallel eps) ===")
    hdr = "ctrl\\dstb  " + "".join(f"{('d'+str(c//1000)+'k' if c != 'dummy' else 'dummy'):>8}" for c in cols)
    print(hdr)
    for i, cs in enumerate(ctrls):
      row = "".join(f"{mat[i, j]:>8.2f}" for j in range(len(cols)))
      print(f"c{cs//1000}k".ljust(11) + row)

  show(succ, "REACH-AVOID SUCCESS  (safe & reached)")
  show(safe, "SAFE RATE  (g>=0 throughout)")

  # worst-case over the real adversaries (exclude dummy column)
  wc_succ = succ[:, :-1].min(axis=1)
  wc_safe = safe[:, :-1].min(axis=1)
  print("\n=== worst-case over archived adversaries (excl. dummy) ===")
  for i, cs in enumerate(ctrls):
    print(f"  c{cs//1000}k: min success={wc_succ[i]:.2f}  min safe={wc_safe[i]:.2f}")
  best = int(np.argmax(wc_succ))
  print(f"\nMAXIMIN controller: c{ctrls[best]//1000}k  "
        f"(worst-case success={wc_succ[best]:.2f}, worst-case safe={wc_safe[best]:.2f})")
  env.close()


if __name__ == "__main__":
  main()
