"""Headless eval-video recorder for a trained policy.

Loads a checkpoint, rolls the policy out in the task's play env, and writes an
mp4 (no interactive viewer) using the same offscreen renderer as training-time
video.  Run headless with::

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_record.py \
        Unitree-Go2-Reach-Avoid \
        --checkpoint-file logs/rsl_rl/go2_reach_avoid/<run>/model_9999.pt \
        --num-envs 6 --env-idx 0 --steps 600 --out /tmp/eval_parkour.mp4

``--terrain`` optionally swaps the scene terrain for the eval (e.g. ``stairs``)
to probe out-of-distribution scenarios.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: str
  num_envs: int = 6
  env_idx: int = 0
  steps: int = 600
  out: str = "/tmp/eval.mp4"
  device: str | None = None
  terrain: Literal["default", "stairs"] = "default"


def _apply_stairs_terrain(env_cfg) -> None:
  """Swap the scene terrain for the rough set (pyramid stairs + slopes); OOD probe."""
  from dataclasses import replace

  from mjlab.terrains.config import ROUGH_TERRAINS_CFG

  tg = replace(ROUGH_TERRAINS_CFG)
  tg.curriculum = False
  env_cfg.scene.terrain.terrain_generator = tg


def run(task_id: str, cfg: EvalConfig) -> None:
  import mjlab.tasks  # noqa: F401  (register tasks)
  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.viewer.env_idx = cfg.env_idx
  if cfg.terrain == "stairs":
    _apply_stairs_terrain(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  out = Path(cfg.out)
  env = VideoRecorder(
    env,
    video_folder=str(out.parent),
    step_trigger=lambda step: step == 0,
    video_length=cfg.steps,
    name_prefix=out.stem,
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(Path(cfg.checkpoint_file).expanduser().resolve()),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  obs, _ = env.reset()
  with torch.no_grad():
    for _ in range(cfg.steps + 5):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
  env.close()
  print(f"[INFO] wrote eval video under {out.parent} (prefix '{out.stem}')")


def main() -> None:
  task_id = sys.argv[1]
  cfg = tyro.cli(EvalConfig, args=sys.argv[2:], prog=f"{sys.argv[0]} {task_id}")
  run(task_id, cfg)


if __name__ == "__main__":
  main()
