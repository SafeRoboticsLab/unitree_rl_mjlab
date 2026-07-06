"""Scenario eval videos for the safety_sb3 chain filter (deployable policy).

Mirrors the rsl_rl eval_rest_video: controlled arrival momentum x gap size,
so the learned stop-vs-jump decision is visible. The filter brakes to a safe
stop whenever it CAN (braking distance v^2/(2a) < nose-distance to the gap);
only an UNSTOPPABLE arrival (close + fast) forces a jump.

  stop_low_momentum   jumpable gap, slow (stoppable)   -> brake to a safe stop
  jump_high_momentum  same gap, fast (unstoppable)     -> jump across
  small_gap           small gap, committed             -> cross
  large_gap           large gap, committed             -> the safety limit

Controlled spawn is done INSIDE the reset event (so the obs reflects it and the
run is deterministic). Run (mjlab conda env; safety_sb3 on sys.path):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_sb3_chain_video.py \
      --run runs_sb3/sb3_chain_parity
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")

import imageio.v2 as imageio  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from safety_sb3 import ReachAvoidPPO  # noqa: E402
from src.isaacs_go2.go2_parkour_isaacs import (  # noqa: E402
  Go2ParkourIsaacsVecEnv,
  rest_margins,
)

_GAP_X = 2.5

# (tag, gap level [0=0.15m .. 9=0.50m], nose-distance to gap, arrival vx)
SCENARIOS = [
  ("stop_low_momentum", 4, 1.5, 1.0),
  ("jump_high_momentum", 4, 0.45, 3.0),
  ("small_gap", 1, 0.4, 2.9),
  ("large_gap", 9, 0.5, 3.2),
]


def _pinned_reset(env, env_ids, dist, vx, asset_cfg=SceneEntityCfg("robot")):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  robot = env.scene[asset_cfg.name]
  o = env.scene.env_origins[env_ids]
  root = robot.data.default_root_state[env_ids].clone()
  n = int(len(env_ids))
  pos = root[:, 0:3] + o
  pos[:, 0] = o[:, 0] + (_GAP_X - 0.35 - dist)
  pos[:, 1] = o[:, 1]
  pos[:, 2] = root[:, 2] + 0.02
  vel = torch.zeros(n, 6, device=env.device)
  vel[:, 0] = vx
  robot.write_root_link_pose_to_sim(
    torch.cat([pos, root[:, 3:7]], dim=-1), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def _scenario_builder(dist, vx):
  def build(play=False):
    from src.tasks.go2_safety_filter.crossing_chain.env_cfg import (
      unitree_go2_crossing_chain_env_cfg,
    )
    cfg = unitree_go2_crossing_chain_env_cfg(play=False)
    cfg.curriculum = {}
    cfg.events.pop("handover_joints", None)
    cfg.events.pop("randomize_terrain", None)
    cfg.events["reset_base"] = EventTermCfg(
      func=_pinned_reset, mode="reset", params={"dist": dist, "vx": vx})
    return cfg
  return build


def record(model, vn, tag, level, dist, vx, steps, device, outdir):
  env = Go2ParkourIsaacsVecEnv(
    1, device, adversary=False, render_mode="rgb_array",
    cfg_builder=_scenario_builder(dist, vx), margin_fn=rest_margins)
  robot = env.mj.scene["robot"]
  terr = env.mj.scene.terrain
  terr.terrain_levels[:] = level
  terr.env_origins[:] = terr.terrain_origins[terr.terrain_levels, terr.terrain_types]
  obs = env.reset()  # pinned_reset runs here -> obs reflects the controlled spawn
  o = env.mj.scene.env_origins
  maxx = float(robot.data.root_link_pos_w[0, 0] - o[0, 0])
  frames = []
  for _ in range(steps):
    no = vn.normalize_obs(obs)
    act, _ = model.predict(no, deterministic=True)
    obs, _r, _d, _i = env.step(act)
    maxx = max(maxx, float(robot.data.root_link_pos_w[0, 0] - o[0, 0]))
    frames.append(np.asarray(env.render()))
  out = os.path.join(outdir, f"chain-{tag}.mp4")
  imageio.mimsave(out, frames, fps=30)
  env.close()
  gap_w = 0.15 + level * (0.35 / 9)
  print(f"[{tag}] gap {gap_w:.2f}m vx={vx} nose_dist={dist} -> "
        f"{'CROSSED' if maxx > _GAP_X + 0.4 else 'STOPPED'} (max x_rel {maxx:.2f})")
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", default="runs_sb3/sb3_chain_parity")
  p.add_argument("--steps", type=int, default=280)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  model = ReachAvoidPPO.load(os.path.join(args.run, "final_model.zip"), device=args.device)
  tmp = Go2ParkourIsaacsVecEnv(1, args.device, adversary=False,
                               cfg_builder=_scenario_builder(1.0, 1.0),
                               margin_fn=rest_margins)
  vn = VecNormalize.load(os.path.join(args.run, "vecnormalize.pkl"), tmp)
  vn.training = False
  outdir = os.path.join(_REPO, "runs_sb3", "chain_eval_videos")
  os.makedirs(outdir, exist_ok=True)
  outs = [record(model, vn, t, lv, d, v, args.steps, args.device, outdir)
          for t, lv, d, v in SCENARIOS]
  print("VIDEOS:", *outs, sep="\n  ")


if __name__ == "__main__":
  main()
