"""Worst-case evaluation matrix for the ISAACS gap-jumping line.

Evaluates a ctrl checkpoint against a battery of disturbances at the game
bound (50 N): no push, random directions, 9 fixed directions, and (when
present in the checkpoint) the learned/archived disturbance actors.

Success per episode = survived to timeout AND ever reached the rest set
(l >= 0) AND never left the safe set (g < 0) — the reach-avoid criterion.

Usage:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_league_worstcase.py \
        --checkpoint logs/rsl_rl/go2_crossing_chain/.../model_XXXX.pt \
        --episodes 512 --num-envs 128
"""

from __future__ import annotations

import argparse
import math
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
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402

from src.tasks.parkour.rl.adversarial_vecenv_wrapper import (  # noqa: E402
  AdversarialReachAvoidVecEnvWrapper,
)
from src.tasks.parkour.rl.isaacs_runner import Go2IsaacsOnPolicyRunner  # noqa: E402

TASK = "Unitree-Go2-Crossing-Chain-ISAACS"
BASELINE = ("logs/rsl_rl/go2_crossing_chain/2026-07-02_23-10-03_chain_handover3/"
            "model_28799.pt")

# 8 horizontal directions + straight down.
_FIXED_DIRS = [
  (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4), 0.0) for k in range(8)
] + [(0.0, 0.0, -1.0)]


def run_opponent(runner, env, policy, opponent, episodes, device):
  """Roll episodes vs one opponent; return success rate.

  opponent: "zero" | "random" | ("dir", (x,y,z)) | ("net", actor_module)
  """
  n = env.num_envs
  robot_env = env.unwrapped
  obs = env.get_observations().to(device)
  succ = 0
  total = 0
  ever_l = torch.zeros(n, dtype=torch.bool, device=device)
  ever_gneg = torch.zeros(n, dtype=torch.bool, device=device)
  while total < episodes:
    with torch.no_grad():
      act = policy(obs)
      if opponent == "zero":
        d = None
      elif opponent == "random":
        d = torch.randn(n, 3, device=device)
      elif opponent[0] == "dir":
        d = torch.tensor(opponent[1], device=device).expand(n, 3).contiguous()
      else:  # ("net", actor)
        d = opponent[1](obs.select("critic"), stochastic_output=False)
      env.set_dstb_action(d)
    obs, g, dones, extras = env.step(act)
    ell = extras["target_margin"]
    ever_l |= ell >= 0
    ever_gneg |= g < 0
    if bool(dones.any()):
      time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
      done_mask = dones.bool()
      s = done_mask & time_outs & ever_l & ~ever_gneg
      succ += int(s.sum())
      total += int(done_mask.sum())
      ever_l = torch.where(done_mask, torch.zeros_like(ever_l), ever_l)
      ever_gneg = torch.where(done_mask, torch.zeros_like(ever_gneg), ever_gneg)
  return succ / max(total, 1)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", default=BASELINE)
  p.add_argument("--episodes", type=int, default=512)
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  dev = args.device
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.isaacs["force_max"] = args.force_max

  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = Go2IsaacsOnPolicyRunner(env, asdict(agent_cfg), device=dev)
  runner.load(args.checkpoint, load_cfg={"actor": True, "critic": True},
              strict=False, map_location=dev)
  policy = runner.get_inference_policy(device=dev)
  wenv: AdversarialReachAvoidVecEnvWrapper = runner.env
  wenv.set_force_scale(1.0)
  wenv.set_rest_edge_clearance(0.3)  # judge with the robustified rest set
  wenv.reset()

  ckpt = torch.load(args.checkpoint, weights_only=False, map_location=dev)
  opponents: list[tuple[str, object]] = [("ZERO", "zero"), ("RANDOM", "random")]
  for i, dvec in enumerate(_FIXED_DIRS):
    name = f"DIR{i}" if i < 8 else "DOWN"
    opponents.append((name, ("dir", dvec)))
  if "dstb_actor_state_dict" in ckpt:
    opponents.append(("LEARNED", ("net", runner.dstb_alg.actor)))

  print(f"=== WORST-CASE MATRIX: {os.path.basename(args.checkpoint)} "
        f"@ {args.force_max:.0f} N, {args.episodes} eps/opponent ===")
  rates = {}
  for name, opp in opponents:
    r = run_opponent(runner, wenv, policy, opp, args.episodes, dev)
    rates[name] = r
    print(f"  {name:8s} success {100*r:5.1f}%")
  worst = min(rates.items(), key=lambda kv: kv[1])
  print(f"  WORST -> {worst[0]} {100*worst[1]:.1f}%")
  env.close()


if __name__ == "__main__":
  main()
