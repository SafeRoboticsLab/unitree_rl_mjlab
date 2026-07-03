"""ISAACS Tier 2: train SB3 IsaacsSAC on the mjlab Go2 vs a push/pull adversary.

Two-player reach-avoid SAC. The control player (12 joint targets) keeps the Go2
upright and tracking its velocity command; the disturbance player (3-D base
force) tries to topple it. Logs scalars to wandb and uploads nominal +
adversarial eval videos for review.

Run (in the ``mjlab`` conda env)::

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
      python scripts/train_isaacs_go2.py --timesteps 300000 --run-name go2-isaacs
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
import wandb  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402
from wandb.integration.sb3 import WandbCallback  # noqa: E402

from safety_sb3 import IsaacsSAC  # noqa: E402
from src.isaacs_go2 import Go2IsaacsEnv, Go2IsaacsVecEnv  # noqa: E402


def policy_action(model, obs_np, *, deterministic=True, use_dstb=True):
  """Concatenated [ctrl, dstb] action from the two ISAACS actors for one obs."""
  obs_t = torch.as_tensor(
    np.asarray(obs_np)[None], device=model.device, dtype=torch.float32
  )
  with torch.no_grad():
    a_ctrl = model.policy.actor(obs_t, deterministic=deterministic).cpu().numpy()[0]
    if use_dstb:
      a_dstb = model.policy.dstb_actor(obs_t, deterministic=deterministic).cpu().numpy()[0]
    else:
      a_dstb = np.zeros(model.policy.dstb_action_dim, dtype=np.float32)
  return np.concatenate([a_ctrl, a_dstb]).astype(np.float32)


class ForceCurriculumCallback(BaseCallback):
  """Anneal the adversary force magnitude force_init -> force_final (5 -> 50 N).

  Mirrors safe_adaptation_dev's linear force curriculum. Writes ``force_max`` on
  every supplied env (train VecEnv + eval envs + the leaderboard eval env) so
  rollouts, video evals, and leaderboard scoring all see the current level.
  """

  def __init__(self, envs, force_init, force_final, curriculum_steps, verbose=0):
    super().__init__(verbose)
    self.envs = [e for e in envs if e is not None]
    self.f0 = float(force_init)
    self.f1 = float(force_final)
    self.steps = max(1, int(curriculum_steps))

  def _cur_force(self):
    frac = min(1.0, self.num_timesteps / self.steps)
    return self.f0 + frac * (self.f1 - self.f0)

  def _on_step(self) -> bool:
    f = self._cur_force()
    for e in self.envs:
      e.force_max = f
    self.logger.record("curriculum/force_N", f)
    return True


class GammaScheduleCallback(BaseCallback):
  """Anneal the reach-avoid discount gamma init -> end (0.95 -> 0.999).

  StepLRMargin schedule from safe_adaptation_dev:
  ``gamma = end - (end - init) * decay ** floor(t / period)`` (clamped at end).
  The period is compressed to fit our (shorter) training budget.
  """

  def __init__(self, init_g, end_g, decay, period, verbose=0):
    super().__init__(verbose)
    self.init_g = float(init_g)
    self.end_g = float(end_g)
    self.decay = float(decay)
    self.period = max(1, int(period))

  def _on_step(self) -> bool:
    n = self.num_timesteps // self.period
    g = self.end_g - (self.end_g - self.init_g) * (self.decay ** n)
    g = min(g, self.end_g)
    self.model.gamma = g
    self.logger.record("curriculum/gamma", g)
    return True


class EntropyFloorCallback(BaseCallback):
  """Clamp the (learned) ctrl & dstb entropy coefficients at min_alpha.

  safe_adaptation_dev flags this as critical: without a floor, learned alpha
  collapses and the policy degenerates. Clamps log_ent_coef in place.
  """

  def __init__(self, min_alpha, verbose=0):
    super().__init__(verbose)
    self.log_min = float(np.log(min_alpha))

  def _on_step(self) -> bool:
    with torch.no_grad():
      for attr in ("log_ent_coef", "dstb_log_ent_coef"):
        t = getattr(self.model, attr, None)
        if t is not None:
          t.clamp_(min=self.log_min)
    return True


class EvalVideoCallback(BaseCallback):
  """Periodic reach-avoid eval with nominal + adversarial videos to wandb."""

  def __init__(self, eval_env, eval_freq, n_episodes, max_steps, video_dir, verbose=1):
    super().__init__(verbose)
    self.eval_env = eval_env
    self.eval_freq = int(eval_freq)
    self.n_episodes = int(n_episodes)
    self.max_steps = int(max_steps)
    self.video_dir = video_dir
    os.makedirs(video_dir, exist_ok=True)
    self._next = self.eval_freq

  def _on_step(self) -> bool:
    if self.num_timesteps >= self._next:
      self._next += self.eval_freq
      for mode, use_dstb in (("nominal", False), ("adversarial", True)):
        self._eval_mode(mode, use_dstb)
    return True

  def _eval_mode(self, mode, use_dstb):
    frames, safe, reached, succ, glens, llens, eplens = [], [], [], [], [], [], []
    for ep in range(self.n_episodes):
      obs, info = self.eval_env.reset()
      ep_safe, ep_reached, ep_len = True, False, 0
      gmin, lmax = info["g_x"], info["l_x"]
      for t in range(self.max_steps):
        action = policy_action(self.model, obs, deterministic=True, use_dstb=use_dstb)
        obs, g, term, trunc, info = self.eval_env.step(action)
        ep_len += 1
        gmin = min(gmin, g)
        lmax = max(lmax, info["l_x"])
        if g < 0.0:
          ep_safe = False
        if info["l_x"] >= 0.0:
          ep_reached = True
        if ep == 0:
          frames.append(np.asarray(self.eval_env.render()))
        if term or trunc:
          break
      safe.append(float(ep_safe))
      reached.append(float(ep_reached))
      succ.append(float(ep_safe and ep_reached))
      glens.append(gmin)
      llens.append(lmax)
      eplens.append(ep_len)

    step = self.num_timesteps
    log = {
      f"eval/{mode}/safe_rate": float(np.mean(safe)),
      f"eval/{mode}/reach_rate": float(np.mean(reached)),
      f"eval/{mode}/success_rate": float(np.mean(succ)),
      f"eval/{mode}/g_min_mean": float(np.mean(glens)),
      f"eval/{mode}/l_max_mean": float(np.mean(llens)),
      f"eval/{mode}/ep_len_mean": float(np.mean(eplens)),
    }
    if frames:
      path = os.path.join(self.video_dir, f"{mode}_{step}.mp4")
      imageio.mimsave(path, frames, fps=50, macro_block_size=1)
      log[f"eval/{mode}/video"] = wandb.Video(path, fps=50, format="mp4")
    wandb.log(log, step=step)
    if self.verbose:
      print(
        f"[eval {mode} @ {step}] success={log[f'eval/{mode}/success_rate']:.2f} "
        f"safe={log[f'eval/{mode}/safe_rate']:.2f} ep_len={log[f'eval/{mode}/ep_len_mean']:.0f}"
      )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--timesteps", type=int, default=3_000_000)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--run-name", default="go2-isaacs")
  p.add_argument("--wandb-project", default="go2-isaacs-tier2")
  p.add_argument("--num-envs", type=int, default=64,
                 help="mjlab GPU-parallel envs (1 -> single gym.Env path)")
  p.add_argument("--train-freq", type=int, default=1)
  p.add_argument("--gradient-steps", type=int, default=16)
  p.add_argument("--batch-size", type=int, default=512)
  p.add_argument("--eval-freq", type=int, default=100_000)
  p.add_argument("--n-eval-episodes", type=int, default=10)
  p.add_argument("--eval-max-steps", type=int, default=400)
  p.add_argument("--ctrl-gain", type=float, default=3.0)
  p.add_argument("--task", default="stabilize", choices=["stabilize", "locomote"],
                 help="stabilize = stance target (zero cmd); locomote = track forward cmd")
  p.add_argument("--cmd-vx", type=float, default=1.0, help="forward command (locomote task)")
  p.add_argument("--learning-rate", type=float, default=1e-4)
  p.add_argument("--learning-starts", type=int, default=20_000)
  # --- faithful fixes (safe_adaptation_dev go2_mujoco_isaacs_v22_long) ---
  p.add_argument("--force-max", type=float, default=50.0, help="final adversary force (N)")
  p.add_argument("--force-init", type=float, default=5.0, help="curriculum start force (N)")
  p.add_argument("--force-curriculum-steps", type=int, default=None,
                 help="anneal force_init->force_max over this many steps (default timesteps/2)")
  p.add_argument("--gamma-init", type=float, default=0.95)
  p.add_argument("--gamma-end", type=float, default=0.999)
  p.add_argument("--gamma-decay", type=float, default=0.1)
  p.add_argument("--gamma-period", type=int, default=None,
                 help="StepLRMargin period (default timesteps/4)")
  p.add_argument("--init-alpha", type=float, default=0.1, help="initial entropy coef")
  p.add_argument("--min-alpha", type=float, default=0.01, help="entropy-coef floor")
  p.add_argument("--save-top-k", type=int, default=20)
  p.add_argument("--use-leaderboard", action="store_true")
  p.add_argument("--leaderboard-freq", type=int, default=150_000)
  p.add_argument("--leaderboard-eval-episodes", type=int, default=2,
                 help="x leaderboard-eval-envs parallel samples per board cell")
  p.add_argument("--leaderboard-eval-envs", type=int, default=32)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  curr_steps = args.force_curriculum_steps or max(1, args.timesteps // 2)
  gamma_period = args.gamma_period or max(1, args.timesteps // 4)

  run = wandb.init(
    project=args.wandb_project, name=args.run_name, config=vars(args),
    sync_tensorboard=True, save_code=True,
  )

  # Envs start at force_init; the curriculum callback ramps force_max each step.
  env_kwargs = dict(device=args.device, force_max=args.force_init, ctrl_gain=args.ctrl_gain,
                    task=args.task, cmd_vx=args.cmd_vx)
  if args.num_envs > 1:
    train_vec = Go2IsaacsVecEnv(num_envs=args.num_envs, render_mode=None, **env_kwargs)
    train_env = VecMonitor(train_vec)
    train_unwrapped = train_vec
  else:
    train_env = Go2IsaacsEnv(render_mode=None, **env_kwargs)
    train_unwrapped = train_env
  eval_env = Go2IsaacsEnv(render_mode="rgb_array", **env_kwargs)

  curriculum_envs = [train_unwrapped, eval_env]
  lb_kwargs = {}
  if args.use_leaderboard:
    # Parallel VecEnv leaderboard scoring (cheap for large save_top_k).
    lb_eval_env = Go2IsaacsVecEnv(
      num_envs=args.leaderboard_eval_envs, render_mode=None, **env_kwargs
    )
    curriculum_envs.append(lb_eval_env)
    lb_kwargs = dict(
      use_leaderboard=True,
      leaderboard_eval_env=lb_eval_env,
      leaderboard_dir=os.path.join("runs", run.id, "leaderboard"),
      leaderboard_freq=args.leaderboard_freq,
      n_eval_episodes=args.leaderboard_eval_episodes,
      save_top_k_ctrl=args.save_top_k,
      save_top_k_dstb=args.save_top_k,
    )

  model = IsaacsSAC(
    "MlpPolicy",
    train_env,
    ctrl_action_dim=Go2IsaacsEnv.CTRL_DIM,
    learning_rate=args.learning_rate,
    buffer_size=1_000_000,
    learning_starts=args.learning_starts,
    batch_size=args.batch_size,
    tau=0.01,
    gamma=args.gamma_init,            # GammaScheduleCallback anneals to gamma_end
    train_freq=(args.train_freq, "step"),
    gradient_steps=args.gradient_steps,
    target_update_interval=2,
    ent_coef=f"auto_{args.init_alpha}",
    ctrl_update_period=2,
    dstb_update_period=2,
    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 512], qf=[256, 256, 256])),
    tensorboard_log=os.path.join("runs", run.id),
    seed=args.seed,
    device=args.device,
    verbose=1,
    **lb_kwargs,
  )

  callbacks = CallbackList(
    [
      WandbCallback(verbose=1),
      ForceCurriculumCallback(curriculum_envs, args.force_init, args.force_max, curr_steps),
      GammaScheduleCallback(args.gamma_init, args.gamma_end, args.gamma_decay, gamma_period),
      EntropyFloorCallback(args.min_alpha),
      EvalVideoCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_episodes=args.n_eval_episodes,
        max_steps=args.eval_max_steps,
        video_dir=os.path.join("runs", run.id, "videos"),
      ),
    ]
  )

  model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
  model.save(os.path.join("runs", run.id, "final_model"))
  run.finish()


if __name__ == "__main__":
  main()
