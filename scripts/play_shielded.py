"""Play Go2 with a safety-shielded walking policy.

Loads two checkpoints:

* ``--walk-checkpoint`` — trained via ``Unitree-Go2-Walking`` (MLP over
  proprioception).
* ``--safe-checkpoint`` — trained via ``Unitree-Go2-Safety-Shield``
  (CNN over proprioception + depth, SafetyPPO).

Usage
-----
    python scripts/play_shielded.py \\
        --walk-checkpoint=logs/rsl_rl/go2_walking_shield/<run>/model_<k>.pt \\
        --safe-checkpoint=logs/rsl_rl/go2_safety_shield/<run>/model_<k>.pt \\
        --threshold=0.05 --num-envs=4
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


WALK_TASK = "Unitree-Go2-Walking"
SAFE_TASK = "Unitree-Go2-Safety-Shield"


@dataclass(frozen=True)
class ShieldedPlayConfig:
  walk_checkpoint: str
  safe_checkpoint: str
  threshold: float = 0.05
  num_envs: int = 4
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  log_every: int = 10
  """Print shield telemetry every N steps."""

def _build_runner(task_id: str, env: RslRlVecEnvWrapper, device: str):
  """Build a runner for a given task and return it (not yet loaded)."""
  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  return runner_cls(env, asdict(agent_cfg), device=device), agent_cfg


def _load_checkpoint(
  runner, path: str, device: str, load_critic: bool = False
) -> None:
  p = Path(path).expanduser().resolve()
  if not p.exists():
    raise FileNotFoundError(f"Checkpoint not found: {p}")
  load_cfg = {"actor": True, "critic": load_critic}
  runner.load(str(p), load_cfg=load_cfg, strict=True, map_location=device)


def run_shielded(cfg: ShieldedPlayConfig) -> None:
  # Registers tasks.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.go2_safety_filter.shield import ShieldedPolicy

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(SAFE_TASK, play=True)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  # --- Walking runner: expects its own agent_cfg structure ---
  walk_runner, _ = _build_runner(WALK_TASK, env, device)
  _load_checkpoint(walk_runner, cfg.walk_checkpoint, device, load_critic=False)
  walk_actor = walk_runner.get_inference_policy(device=device)

  # --- Safety runner: CNN actor + critic trained via SafetyPPO ---
  safe_runner, _ = _build_runner(SAFE_TASK, env, device)
  _load_checkpoint(safe_runner, cfg.safe_checkpoint, device, load_critic=True)
  safe_actor = safe_runner.get_inference_policy(device=device)

  safe_runner.alg.eval_mode()
  safe_critic = safe_runner.alg.critic

  # --- Shielded policy ---
  step_counter = {"i": 0}

  def _log(stats):
    i = step_counter["i"]
    if i % cfg.log_every == 0:
      print(
        f"[shield] step={i:5d} | "
        f"V_safe: mean={stats.v_safe_mean:+.3f} min={stats.v_safe_min:+.3f} | "
        f"safe_rate={stats.safe_rate:.2f} override_rate={stats.override_rate:.2f}",
        flush=True,
      )
    step_counter["i"] += 1

  policy = ShieldedPolicy(
    walk_actor=walk_actor,
    safe_actor=safe_actor,
    safe_critic=safe_critic,
    threshold=cfg.threshold,
    log_fn=_log,
  )

  # --- Viewer ---
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    viewer = NativeMujocoViewer(env, policy)
  elif resolved_viewer == "viser":
    viewer = ViserPlayViewer(env, policy)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  viewer.run()
  env.close()


def main():
  cfg = tyro.cli(ShieldedPlayConfig, prog=sys.argv[0])
  run_shielded(cfg)


if __name__ == "__main__":
  main()
