"""Record review videos of the ISAACS-robustified policy under disturbances.

One opponent per run:
    --opponent zero            no disturbance (nominal brake/jump behavior)
    --opponent dir --dir-vec X Y Z    sustained 50 N push in a fixed direction
    --opponent learned         the trained adversary (deterministic)
    --opponent random          random-direction push

Records env 0 with the side-view camera; prints per-episode outcomes
(SAFE REST / FELL) with reach-avoid judging (ever l>=0, never g<0).

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_isaacs_video.py \
        --checkpoint <model.pt> --opponent dir --dir-vec 0 -1 0 --tag dir6
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

from src.tasks.parkour.rl.isaacs_runner import Go2IsaacsOnPolicyRunner  # noqa: E402

TASK = "Unitree-Go2-Crossing-Chain-ISAACS"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--opponent", default="zero",
                 choices=["zero", "random", "dir", "learned"])
  p.add_argument("--dir-vec", type=float, nargs=3, default=[0.0, -1.0, 0.0])
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--steps", type=int, default=1300)
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--tag", default="clip")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  dev = args.device
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.viewer.distance = 4.5
  env_cfg.viewer.elevation = -20.0
  env_cfg.viewer.azimuth = 90.0
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.isaacs["force_max"] = args.force_max

  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev, render_mode="rgb_array")
  vid_dir = os.path.join(_REPO, "logs", "isaacs_eval")
  os.makedirs(vid_dir, exist_ok=True)
  env = VideoRecorder(env, video_folder=vid_dir, step_trigger=lambda s: s == 0,
                      video_length=args.steps, disable_logger=True,
                      name_prefix=f"isaacs-{args.tag}")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = Go2IsaacsOnPolicyRunner(env, asdict(agent_cfg), device=dev)
  runner.load(args.checkpoint, load_cfg={"actor": True, "critic": True},
              strict=False, map_location=dev)
  policy = runner.get_inference_policy(device=dev)
  wenv = runner.env
  wenv.set_force_scale(1.0)
  wenv.set_rest_edge_clearance(0.3)
  wenv.reset()
  obs, _ = wenv.reset()

  n = args.num_envs
  robot = env.unwrapped.scene["robot"]
  ever_l = torch.zeros(n, dtype=torch.bool, device=dev)
  ever_g = torch.zeros(n, dtype=torch.bool, device=dev)
  dvec = torch.tensor(args.dir_vec, device=dev)
  ep = 1
  t0 = 0
  for t in range(args.steps):
    with torch.no_grad():
      act = policy(obs)
      if args.opponent == "zero":
        d = None
      elif args.opponent == "random":
        d = torch.randn(n, 3, device=dev)
      elif args.opponent == "dir":
        d = dvec.expand(n, 3).contiguous()
      else:
        d = runner.dstb_alg.actor(obs.select("critic"), stochastic_output=False)
      wenv.set_dstb_action(d)
    obs, g, dones, extras = wenv.step(act)
    ever_l |= extras["target_margin"] >= 0
    ever_g |= g < 0
    if bool(dones[0]):
      t_o = bool(extras.get("time_outs", torch.zeros_like(dones))[0])
      outcome = ("SAFE REST" if (t_o and ever_l[0] and not ever_g[0])
                 else "SURVIVED (no rest)" if t_o else "FELL")
      print(f"[env0] episode {ep}: steps {t0}-{t} ({t0*0.02:.1f}-{t*0.02:.1f}s) {outcome}")
      ep += 1
      t0 = t + 1
      ever_l[0] = False
      ever_g[0] = False
  print(f"video -> {vid_dir}/isaacs-{args.tag}-step-0.mp4")
  env.close()


if __name__ == "__main__":
  main()
