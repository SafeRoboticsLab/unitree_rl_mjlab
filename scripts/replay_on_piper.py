"""Replay a pretrained Go2-only walking policy on the Go2+Piper+payload env.

The new env has identical proprioceptive observation and action spaces to
the Go2-only env (the Piper arm's joints are held by XML actuators but are
excluded from the policy's obs/action space), so a Go2-only checkpoint
can be dropped straight in.

Usage
-----
  python scripts/replay_on_piper.py \
      --task Unitree-Go2-Piper-Flat \
      --checkpoint logs/rsl_rl/go2_velocity/2026-03-26_22-58-36/model_1000.pt \
      --num-envs 16 --steps 500

Add ``--viewer`` to launch a MuJoCo viewer (requires DISPLAY).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch


def _load_checkpoint_remapped(runner, ckpt_path: str) -> None:
  """Load a legacy Go2 checkpoint. Old MLPModel kept its std under
  ``distribution.std_param``; the current version stores it at ``std``.
  Remap on the fly so strict loading still succeeds."""
  loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  actor_sd = loaded["actor_state_dict"]
  if "distribution.std_param" in actor_sd:
    actor_sd["std"] = actor_sd.pop("distribution.std_param")
  missing, unexpected = runner.alg.actor.load_state_dict(actor_sd, strict=False)
  if missing or unexpected:
    raise RuntimeError(
      f"State-dict remap incomplete: missing={missing}, unexpected={unexpected}"
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Unitree-Go2-Piper-Flat")
  parser.add_argument(
    "--checkpoint",
    default="logs/rsl_rl/go2_velocity/2026-03-26_22-58-36/model_1000.pt",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--viewer", action="store_true")
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")

  import mjlab.tasks  # noqa: F401 — populates registry
  import src.tasks  # noqa: F401 — registers Unitree-Go2-Piper-*
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env_w, asdict(agent_cfg), device=args.device)

  ckpt = Path(args.checkpoint)
  if not ckpt.exists():
    raise FileNotFoundError(ckpt)
  _load_checkpoint_remapped(runner, str(ckpt))
  policy = runner.get_inference_policy(device=args.device)
  print(f"[INFO] Loaded checkpoint: {ckpt}")

  if args.viewer:
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(env_w, policy).run()
    env.close()
    return

  # Headless rollout: collect stability statistics.
  obs = env_w.get_observations()
  total_term = 0
  root_z_trajectory = []
  for _ in range(args.steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = env_w.step(action)
    root_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    root_z_trajectory.append(root_z.mean().item())
    total_term += int(dones.sum().item())

  print(
    f"[RESULT] {args.num_envs} envs x {args.steps} steps | "
    f"mean root-z traj start / mid / end: "
    f"{root_z_trajectory[0]:.3f} / "
    f"{root_z_trajectory[len(root_z_trajectory)//2]:.3f} / "
    f"{root_z_trajectory[-1]:.3f}"
  )
  print(
    f"[RESULT] total terminations: {total_term} "
    f"({total_term / (args.num_envs * args.steps):.1%} of env-steps)"
  )
  env.close()


if __name__ == "__main__":
  main()
