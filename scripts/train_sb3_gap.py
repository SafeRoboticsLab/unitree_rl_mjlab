"""safety_sb3 gap reach-avoid training (Phase 1: single-agent ReachAvoidPPO).

Trains the on-policy safety_sb3 ReachAvoidPPO over the mjlab Go2 gap env via the
Go2ParkourIsaacsVecEnv bridge (adversary OFF -> ctrl-only action space; g is the
reward channel, l_x rides in info). This is the SB3 replacement for the rsl_rl
Unitree-Go2-Gap-ReachAvoid training, on the release codebase.

Run (mjlab conda env -- has mjlab + stable_baselines3; safety_sb3 on sys.path):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/train_sb3_gap.py \
        --num-envs 512 --steps 3000000
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")

from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from safety_sb3 import ReachAvoidPPO  # noqa: E402
from src.isaacs_go2 import Go2ParkourIsaacsVecEnv  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--steps", type=int, default=3_000_000)
  p.add_argument("--n-steps", type=int, default=48, help="rollout length per env")
  p.add_argument("--batch-size", type=int, default=8192)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--tag", default="sb3_gap_ra")
  args = p.parse_args()

  env = Go2ParkourIsaacsVecEnv(
    num_envs=args.num_envs, device=args.device, adversary=False)
  env = VecMonitor(env)

  model = ReachAvoidPPO(
    "MlpPolicy", env,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=args.lr,
    ent_coef=0.005,
    clip_range=0.2,
    max_grad_norm=1.0,
    policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    tensorboard_log=os.path.join(_REPO, "runs_sb3", args.tag),
    verbose=1,
    device=args.device,
  )
  model.learn(total_timesteps=args.steps, progress_bar=False)
  out = os.path.join(_REPO, "runs_sb3", args.tag, "final_model.zip")
  model.save(out)
  print(f"[done] saved {out}")


if __name__ == "__main__":
  main()
