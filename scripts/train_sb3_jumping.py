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

from stable_baselines3.common.callbacks import (  # noqa: E402
  BaseCallback,
  CallbackList,
  CheckpointCallback,
)
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402


class ForceRampCallback(BaseCallback):
  """Ramp the adversary's force magnitude from weak to force_max over training,
  so the ctrl policy adapts to a strengthening attacker instead of collapsing
  under full 50 N from step one (the ISAACS treadmill: ep_len 192 -> 20)."""

  def __init__(self, force_max, ramp_steps, force_start=8.0):
    super().__init__()
    self.fmax, self.ramp, self.fstart = force_max, ramp_steps, force_start
    self._bridge = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e

  def _on_step(self):
    frac = min(1.0, self.num_timesteps / max(1, self.ramp))
    self._bridge.force_max = self.fstart + (self.fmax - self.fstart) * frac
    return True


class VideoWandbCallback(BaseCallback):
  """Periodically roll out the current (deterministic) policy in a small
  render env and upload the clip to wandb as eval/video -- mirrors the rsl_rl
  train/video behavior so every run has watchable eval footage."""

  def __init__(self, eval_env_fn, interval, video_len=200):
    super().__init__()
    self.eval_env_fn, self.interval, self.video_len = eval_env_fn, interval, video_len
    self._env = None
    self._last = 0

  def _on_training_start(self):
    self._env = self.eval_env_fn()  # raw bridge, adversary off, render on

  def _on_step(self):
    if self.num_timesteps - self._last >= self.interval:
      self._last = self.num_timesteps
      self._log_video()
    return True

  def _log_video(self):
    import numpy as np
    import wandb
    vn = self.model.get_vec_normalize_env()
    obs = self._env.reset()
    frames = []
    for _ in range(self.video_len):
      o = vn.normalize_obs(obs) if vn is not None else obs
      act, _ = self.model.predict(o, deterministic=True)
      obs, _r, _d, _i = self._env.step(act)
      frames.append(np.asarray(self._env.render()))
    vid = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W) uint8
    wandb.log({"eval/video": wandb.Video(vid, fps=30, format="mp4")},
              step=self.num_timesteps)

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
  p.add_argument("--dstb-pretrain", type=int, default=20,
                 help="dstb-pretrain ROLLOUTS (not rsl_rl iters); keep << total rollouts")
  p.add_argument("--load", default=None, help="warm-start checkpoint (.zip)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--tag", default=None)
  p.add_argument("--wandb-project", default="safety_sb3_gap")
  p.add_argument("--no-wandb", action="store_true")
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
      # Schedule is in ROLLOUTS (collect+update), NOT rsl_rl iterations. At
      # n_envs*n_steps per rollout the whole run is only steps/(n_envs*n_steps)
      # rollouts, so dstb_pretrain must be a small fraction (400 rollouts would
      # never finish). ctrl warm-started -> short pretrain to bootstrap dstb.
      akw = dict(ctrl_action_dim=CTRL_DIM,
                 dstb_pretrain_rollouts=args.dstb_pretrain,
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
      inner = env  # the (possibly higher-dim-action) env to warm-start onto
      env = VecNormalize.load(vn_path, env)
      env.training = True
      # VecNormalize.load restores the SAVED spaces; when warm-starting the
      # 15-dim adversary (ctrl+dstb) from a 12-dim single-agent checkpoint the
      # action space must stay the new env's (only obs stats transfer).
      env.action_space = inner.action_space
      env.observation_space = inner.observation_space
      common = {k: v for k, v in common.items() if k != "normalize_obs"}
    model = Algo("MlpPolicy", env, **akw, **common)
    # exact_match=False so IsaacsPPO warm-starts ONLY its ctrl player
    # (self.policy) from a single-agent chain checkpoint; the dstb player and
    # leaderboard stay fresh (the rsl_rl ISAACS ctrl-from-model_28799 pattern).
    model.set_parameters(args.load, exact_match=False, device=args.device)
    print(f"[warm-start] loaded {args.load} (ctrl policy)")
  else:
    model = Algo("MlpPolicy", env, **akw, **common)

  # Periodic checkpoints (+ VecNormalize) so runs are evaluable mid-training.
  ckpt_cb = CheckpointCallback(
    save_freq=max(1, 2_000_000 // args.num_envs),
    save_path=os.path.join(_REPO, "runs_sb3", tag, "checkpoints"),
    name_prefix="model", save_vecnormalize=True)
  cbs = [ckpt_cb]
  if args.stage == "chain" and args.adversary:
    # force ramp to force_max over ~55% of training, then hold.
    cbs.append(ForceRampCallback(args.force_max, int(0.55 * args.steps)))

  wb_run = None
  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wb_run = wandb.init(project=args.wandb_project, name=tag,
                        config=vars(args), sync_tensorboard=True,
                        save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    # eval-video env: small, render on, adversary OFF (show the ctrl policy).
    if args.stage == "chain":
      eval_fn = lambda: make_chain_vecenv(2, args.device, adversary=False,
                                          render_mode="rgb_array")
    elif args.stage == "landing":
      eval_fn = lambda: make_landing_vecenv(2, args.device, render_mode="rgb_array")
    else:
      eval_fn = lambda: make_crossing_vecenv(2, args.device, render_mode="rgb_array")
    cbs.append(VideoWandbCallback(eval_fn, interval=max(1, args.steps // 8)))

  model.learn(total_timesteps=args.steps, progress_bar=False,
              callback=CallbackList(cbs))
  if wb_run is not None:
    wb_run.finish()
  outdir = os.path.join(_REPO, "runs_sb3", tag)
  model.save(os.path.join(outdir, "final_model.zip"))
  vn = model.get_vec_normalize_env()
  if vn is not None:
    vn.save(os.path.join(outdir, "vecnormalize.pkl"))
  print(f"[done] saved {outdir}/final_model.zip (+ vecnormalize)")


if __name__ == "__main__":
  main()
