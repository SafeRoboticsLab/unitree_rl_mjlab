"""Stop-vs-crawl decision boundary benchmark for the crawl safety task.

Sweeps bar clearance x arrival speed (plus a distance sweep at high speed
through the must-crawl window) with controlled spawns, and classifies each
episode:

  STOPPED  — reached rest (l >= 0) before the bar, never unsafe (only possible
             on IMPOSSIBLE rows: passable rows exclude the approach from the
             rest set, so stopping there never satisfies l)
  CRAWLED  — passed the bar and reached rest beyond it, never unsafe
  STRUCK   — ever g < 0 (bar strike / fall), or terminated early
  PARKED   — survived to horizon stationary before the bar without resting
             counting (safe but non-live on a passable row)
  UNSET    — survived to horizon without rest, moving / past the bar

Headline science: the learned 50% stop-vs-crawl frontier vs the analytic
curves — crouch feasibility floor (clearance ~0.22 m) and the brakeability
boundary d = v^2/(2a).

Usage:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/eval_crawl_boundary.py \
        --checkpoint logs/rsl_rl/go2_crawl/.../model_XXXX.pt --num-envs 64
"""

from __future__ import annotations

import argparse
import csv
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

from src.isaacs_go2.crawl_filter_terrain import (  # noqa: E402
  _BAR_X,
  BAR_DEPTH,
  ROW_CLEARANCES,
)
from src.tasks.parkour.rl.reach_avoid_runner import (  # noqa: E402
  ParkourReachRestOnPolicyRunner,
)

TASK = "Unitree-Go2-Crawl"
_NOSE = 0.35
_A_BRAKE = 3.0

# Main grid: rows spanning trivial -> floor -> impossible, at a stoppable
# approach distance (the pure clearance-vs-speed decision).
GRID_ROWS = (0, 2, 4, 7, 8, 9)          # 0.35 0.31 0.27 0.22 0.18 0.15
GRID_SPEEDS = (0.5, 1.5, 2.5)
GRID_D = 1.8

# Distance sweep at v=2.5 (brakeability boundary v^2/6 = 1.04): spans
# DOOMED-ish -> MUST-CRAWL -> stoppable, on a passable and an impossible row.
DSWEEP_V = 2.5
DSWEEP_D = (0.6, 0.8, 1.0, 1.2, 1.5)
DSWEEP_ROWS = (4, 9)                    # 0.27 (passable) / 0.15 (impossible)


def pin_row(env, row: int) -> None:
  terrain = env.unwrapped.scene.terrain
  terrain.terrain_levels[:] = int(row)
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]


def run_cell(wenv, policy, row: int, d: float, v: float, episodes: int, dev: str):
  """Roll `episodes` controlled spawns; return outcome counts."""
  raw = wenv.unwrapped
  robot = raw.scene["robot"]
  n = raw.num_envs
  horizon = int(raw.max_episode_length) - 2
  counts = {"STOPPED": 0, "CRAWLED": 0, "STRUCK": 0, "PARKED": 0, "UNSET": 0}

  done_total = 0
  # Controlled spawns happen INSIDE the reset event (reset_takeover_crawl's
  # _eval_spawn hook): out-of-band teleports leave staged sim writes and stale
  # raycast caches that misfire terrain-relative terminations for a step.
  raw._eval_spawn = {"d": d, "v": v}
  while done_total < episodes:
    pin_row(raw, row)
    wenv.reset()
    origins = raw.scene.env_origins
    obs = wenv.get_observations().to(dev)

    active = torch.ones(n, dtype=torch.bool, device=dev)
    ever_gneg = torch.zeros(n, dtype=torch.bool, device=dev)
    rested = torch.zeros(n, dtype=torch.bool, device=dev)
    rest_x = torch.full((n,), float("nan"), device=dev)

    for _ in range(horizon):
      with torch.no_grad():
        act = policy(obs)
      obs, g, dones, extras = wenv.step(act)
      x_rel = robot.data.root_link_pos_w[:, 0] - origins[:, 0]
      ever_gneg |= active & (g < 0)
      newly_rested = active & ~rested & (extras["target_margin"] >= 0)
      rest_x = torch.where(newly_rested, x_rel, rest_x)
      rested |= newly_rested
      if bool(dones.any()):
        t_o = extras.get("time_outs", torch.zeros_like(dones)).bool()
        ended = active & dones.bool() & ~t_o
        # early termination (fall/strike) counts as STRUCK
        ever_gneg |= ended
        active &= ~dones.bool()
      if not bool(active.any()):
        break

    crossed_at_rest = rest_x > (_BAR_X + BAR_DEPTH)
    x_final = robot.data.root_link_pos_w[:, 0] - origins[:, 0]
    spd_final = robot.data.root_link_lin_vel_w[:, :2].norm(dim=1)
    for i in range(n):
      if done_total >= episodes:
        break
      if bool(ever_gneg[i]):
        counts["STRUCK"] += 1
      elif bool(rested[i]) and bool(crossed_at_rest[i]):
        counts["CRAWLED"] += 1
      elif bool(rested[i]):
        counts["STOPPED"] += 1
      elif (bool(active[i]) and float(x_final[i]) < _BAR_X
            and float(spd_final[i]) < 0.5):
        counts["PARKED"] += 1
      else:
        counts["UNSET"] += 1
      done_total += 1
  return counts


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--episodes", type=int, default=64, help="episodes per cell")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None, help="CSV path (default: alongside ckpt)")
  args = p.parse_args()
  dev = args.device

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  # No curricula in eval: crawl_filter_levels would demote/re-roll the pinned
  # terrain rows (and their env origins) on every reset.
  env_cfg.curriculum = {}
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=dev)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = ParkourReachRestOnPolicyRunner(env, asdict(agent_cfg), device=dev)
  runner.load(args.checkpoint, load_cfg={"actor": True, "critic": True},
              strict=False, map_location=dev)
  policy = runner.get_inference_policy(device=dev)
  wenv = runner.env
  wenv.reset()

  rows_out = []
  tag = os.path.basename(args.checkpoint).replace(".pt", "")
  print(f"=== CRAWL DECISION BOUNDARY: {tag}, {args.episodes} eps/cell ===")
  print(f"--- main grid (d = {GRID_D} m, stoppable approach) ---")
  print(f"{'clr':>6} | " + " | ".join(f"v={v:<4}" for v in GRID_SPEEDS))
  for row in GRID_ROWS:
    clr = ROW_CLEARANCES[row]
    cells = []
    for v in GRID_SPEEDS:
      c = run_cell(wenv, policy, row, GRID_D, v, args.episodes, dev)
      tot = sum(c.values())
      cells.append(c)
      rows_out.append({"sweep": "grid", "row": row, "clearance": clr,
                       "d": GRID_D, "v": v, **c, "total": tot})
    def cell_str(c):
      tot = max(1, sum(c.values()))
      return (f"S{100*c['STOPPED']//tot:3d} C{100*c['CRAWLED']//tot:3d} "
              f"X{100*c['STRUCK']//tot:3d} P{100*c['PARKED']//tot:3d}")
    print(f"{clr:6.3f} | " + " | ".join(cell_str(c) for c in cells))

  print(f"--- d-sweep @ v = {DSWEEP_V} (brake boundary d = {DSWEEP_V**2/6:.2f} m) ---")
  for row in DSWEEP_ROWS:
    clr = ROW_CLEARANCES[row]
    for d in DSWEEP_D:
      c = run_cell(wenv, policy, row, d, DSWEEP_V, args.episodes, dev)
      rows_out.append({"sweep": "dsweep", "row": row, "clearance": clr,
                       "d": d, "v": DSWEEP_V, **c, "total": sum(c.values())})
      tot = max(1, sum(c.values()))
      print(f"  clr {clr:.3f} d {d:.1f}: STOP {100*c['STOPPED']/tot:5.1f}% "
            f"CRAWL {100*c['CRAWLED']/tot:5.1f}% STRUCK {100*c['STRUCK']/tot:5.1f}% "
            f"PARKED {100*c['PARKED']/tot:5.1f}% UNSET {100*c['UNSET']/tot:5.1f}%")

  out = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                 f"boundary_{tag}.csv")
  with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
  print(f"[csv] {out}")
  env.close()


if __name__ == "__main__":
  main()
