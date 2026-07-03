"""ISAACS forced-crossing: train SB3 IsaacsSAC on the trapped-island Go2 task.

The robot must cross a (curriculum-widening) gap to reach the robustly-safe far
platform, because the ISAACS adversary makes staying on the small island unsafe.
Skill + safety-decision co-emerge under reach-avoid + adversary — no warm-start,
no reward shaping. Reuses the Tier-2 callbacks + faithful hyperparameters.

Eval ``success_rate`` = safe (never fell) AND reached (crossed to far platform)
= the "cross-to-safety under a non-negatable push" behavior.

Run::

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/train_isaacs_island.py \
        --num-envs 128 --gradient-steps 32 --use-leaderboard --run-name go2-island
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402
from wandb.integration.sb3 import WandbCallback  # noqa: E402

from safety_sb3 import IsaacsSAC  # noqa: E402
from src.isaacs_go2 import Go2IslandCrossingEnv, Go2IslandCrossingVecEnv  # noqa: E402
from train_isaacs_go2 import (  # noqa: E402
  EntropyFloorCallback,
  EvalVideoCallback,
  ForceCurriculumCallback,
  GammaScheduleCallback,
)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--timesteps", type=int, default=3_000_000)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--run-name", default="go2-island-crossing")
  p.add_argument("--wandb-project", default="go2-isaacs-tier3")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--gradient-steps", type=int, default=32)
  p.add_argument("--batch-size", type=int, default=512)
  p.add_argument("--eval-freq", type=int, default=100_000)
  p.add_argument("--n-eval-episodes", type=int, default=10)
  p.add_argument("--eval-max-steps", type=int, default=300)  # 6 s @ 50 Hz
  p.add_argument("--ctrl-gain", type=float, default=3.0)
  p.add_argument("--learning-rate", type=float, default=1e-4)
  p.add_argument("--learning-starts", type=int, default=20_000)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--force-init", type=float, default=15.0)  # real pressure from start
  p.add_argument("--force-curriculum-steps", type=int, default=None)
  p.add_argument("--gamma-init", type=float, default=0.95)
  p.add_argument("--gamma-end", type=float, default=0.999)
  p.add_argument("--gamma-decay", type=float, default=0.1)
  p.add_argument("--gamma-period", type=int, default=None)
  p.add_argument("--init-alpha", type=float, default=0.1)
  p.add_argument("--min-alpha", type=float, default=0.01)
  p.add_argument("--save-top-k", type=int, default=20)
  p.add_argument("--use-leaderboard", action="store_true")
  p.add_argument("--leaderboard-freq", type=int, default=150_000)
  p.add_argument("--leaderboard-eval-episodes", type=int, default=2)
  p.add_argument("--leaderboard-eval-envs", type=int, default=32)
  p.add_argument("--train-reset-mode", default="mix", choices=["mix", "island", "midair_land"])
  p.add_argument("--eval-reset-mode", default="island", choices=["mix", "island", "midair_land"])
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  curr_steps = args.force_curriculum_steps or max(1, args.timesteps // 2)
  gamma_period = args.gamma_period or max(1, args.timesteps // 4)

  run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                   sync_tensorboard=True, save_code=True)

  env_kwargs = dict(device=args.device, force_max=args.force_init, ctrl_gain=args.ctrl_gain)
  train_vec = Go2IslandCrossingVecEnv(num_envs=args.num_envs, render_mode=None,
                                      reset_mode=args.train_reset_mode, **env_kwargs)
  train_env = VecMonitor(train_vec)
  eval_env = Go2IslandCrossingEnv(render_mode="rgb_array",
                                  reset_mode=args.eval_reset_mode, **env_kwargs)
  curriculum_envs = [train_vec, eval_env]

  lb_kwargs = {}
  if args.use_leaderboard:
    lb_eval_env = Go2IslandCrossingVecEnv(num_envs=args.leaderboard_eval_envs, render_mode=None,
                                          reset_mode=args.train_reset_mode, **env_kwargs)
    curriculum_envs.append(lb_eval_env)
    lb_kwargs = dict(
      use_leaderboard=True, leaderboard_eval_env=lb_eval_env,
      leaderboard_dir=os.path.join("runs", run.id, "leaderboard"),
      leaderboard_freq=args.leaderboard_freq, n_eval_episodes=args.leaderboard_eval_episodes,
      save_top_k_ctrl=args.save_top_k, save_top_k_dstb=args.save_top_k,
    )

  model = IsaacsSAC(
    "MlpPolicy", train_env, ctrl_action_dim=Go2IslandCrossingEnv.CTRL_DIM,
    learning_rate=args.learning_rate, buffer_size=1_000_000,
    learning_starts=args.learning_starts, batch_size=args.batch_size, tau=0.01,
    gamma=args.gamma_init, train_freq=(1, "step"), gradient_steps=args.gradient_steps,
    target_update_interval=2, ent_coef=f"auto_{args.init_alpha}",
    ctrl_update_period=2, dstb_update_period=2,
    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 512], qf=[256, 256, 256])),
    tensorboard_log=os.path.join("runs", run.id), seed=args.seed, device=args.device,
    verbose=1, **lb_kwargs,
  )

  callbacks = CallbackList([
    WandbCallback(verbose=1),
    ForceCurriculumCallback(curriculum_envs, args.force_init, args.force_max, curr_steps),
    GammaScheduleCallback(args.gamma_init, args.gamma_end, args.gamma_decay, gamma_period),
    EntropyFloorCallback(args.min_alpha),
    EvalVideoCallback(eval_env, eval_freq=args.eval_freq, n_episodes=args.n_eval_episodes,
                      max_steps=args.eval_max_steps, video_dir=os.path.join("runs", run.id, "videos")),
  ])

  model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
  model.save(os.path.join("runs", run.id, "final_model"))
  run.finish()


if __name__ == "__main__":
  main()
