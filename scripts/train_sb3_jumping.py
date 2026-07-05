"""safety_sb3 gap-jumping pipeline (Phase 2): landing -> crossing -> chain.

Mirrors the rsl_rl pipeline on the release codebase. Each stage wraps the
corresponding mjlab env cfg (spawn strata + reverse curricula + handover are
reused as-is) through the isaacs_go2 SB3 bridge, trained with a safety_sb3
learner:

  landing   avoid-only  SafetyPPO      mid-air-over-gap -> soft land
  crossing  avoid-only  SafetyPPO      reverse curriculum launch->land
  chain     reach-avoid ReachAvoidPPO  arrival momentum -> safe rest
  chain --adversary     IsaacsPPO      two-player ISAACS game

Warm-start a stage from the previous one with --load. Run in the mjlab conda
env (mjlab + stable_baselines3; safety_sb3 on sys.path), MUJOCO_GL=egl.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/train_sb3_jumping.py \
        --stage landing --num-envs 2048 --steps 8000000
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")

from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from safety_sb3 import IsaacsPPO, ReachAvoidPPO, SafetyPPO  # noqa: E402
from src.isaacs_go2.go2_parkour_isaacs import (  # noqa: E402
  CTRL_DIM,
  make_chain_vecenv,
  make_crossing_vecenv,
  make_landing_vecenv,
)

_NET = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--stage", required=True, choices=["landing", "crossing", "chain"])
  p.add_argument("--adversary", action="store_true", help="chain: two-player IsaacsPPO")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--steps", type=int, default=8_000_000)
  p.add_argument("--n-steps", type=int, default=48)
  p.add_argument("--batch-size", type=int, default=16384)
  p.add_argument("--lr", type=float, default=5e-4)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--load", default=None, help="warm-start checkpoint (.zip)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--tag", default=None)
  args = p.parse_args()

  tag = args.tag or f"sb3_{args.stage}{'_isaacs' if args.adversary else ''}"
  # rsl_rl-parity recipe (the config that reached ep_len 135 on landing):
  #   obs normalization (both nets), low entropy 1e-4, init action std 0.3
  #   (log_std_init=ln 0.3), KL-adaptive LR (desired_kl 0.01, lr 5e-4).
  policy_kwargs = dict(log_std_init=math.log(0.3), **_NET)
  common = dict(gamma=0.99, gae_lambda=0.95, learning_rate=args.lr,
                ent_coef=1e-4, clip_range=0.2, max_grad_norm=1.0, n_epochs=5,
                n_steps=args.n_steps, batch_size=args.batch_size,
                normalize_obs=True, adaptive_lr=True, desired_kl=0.01,
                policy_kwargs=policy_kwargs, verbose=1, device=args.device,
                tensorboard_log=os.path.join(_REPO, "runs_sb3", tag))

  if args.stage == "landing":
    env = VecMonitor(make_landing_vecenv(args.num_envs, args.device))
    Algo, akw = SafetyPPO, {}
  elif args.stage == "crossing":
    env = VecMonitor(make_crossing_vecenv(args.num_envs, args.device))
    Algo, akw = SafetyPPO, {}
  else:  # chain
    env = VecMonitor(make_chain_vecenv(
      args.num_envs, args.device, adversary=args.adversary, force_max=args.force_max))
    if args.adversary:
      Algo = IsaacsPPO
      akw = dict(ctrl_action_dim=CTRL_DIM, dstb_pretrain_rollouts=400,
                 ctrl_rollouts_per_cycle=12, dstb_rollouts_per_cycle=3,
                 use_leaderboard=True,
                 leaderboard_dir=os.path.join(_REPO, "runs_sb3", tag, "leaderboard"))
    else:
      Algo, akw = ReachAvoidPPO, {}

  if args.load:
    # Restore the prior stage's obs-normalization stats onto this env, then
    # load the policy (SB3 VecNormalize stats live outside the .zip).
    from stable_baselines3.common.vec_env import VecNormalize
    vn_path = args.load.replace("final_model.zip", "vecnormalize.pkl")
    if os.path.exists(vn_path):
      env = VecNormalize.load(vn_path, env)
      env.training = True
      common = {k: v for k, v in common.items() if k != "normalize_obs"}
    model = Algo("MlpPolicy", env, **akw, **common)
    model.set_parameters(args.load, device=args.device)
    print(f"[warm-start] loaded {args.load}")
  else:
    model = Algo("MlpPolicy", env, **akw, **common)

  model.learn(total_timesteps=args.steps, progress_bar=False)
  outdir = os.path.join(_REPO, "runs_sb3", tag)
  model.save(os.path.join(outdir, "final_model.zip"))
  vn = model.get_vec_normalize_env()
  if vn is not None:
    vn.save(os.path.join(outdir, "vecnormalize.pkl"))
  print(f"[done] saved {outdir}/final_model.zip (+ vecnormalize)")


if __name__ == "__main__":
  main()
