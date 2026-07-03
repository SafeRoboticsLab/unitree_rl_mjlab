"""Smoke-probe the Go2IsaacsEnv: shapes, g/l ranges, force effect, render."""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safety-stable-baselines")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.isaacs_go2 import Go2IsaacsEnv  # noqa: E402

env = Go2IsaacsEnv(device="cuda:0", render_mode="rgb_array")
print("obs_space", env.observation_space.shape, "act_space", env.action_space.shape)

obs, info = env.reset()
print("reset obs", obs.shape, "dtype", obs.dtype)
print("reset info g/l", round(info["g_x"], 3), round(info["l_x"], 3),
      "base_z", round(info["base_z"], 3), "up", round(info["uprightness"], 3))

# Hold default pose (zero ctrl), no disturbance: should stay safe (g>0) a while.
gs = []
for i in range(40):
  a = np.zeros(env.action_space.shape, dtype=np.float32)
  obs, g, term, trunc, info = env.step(a)
  gs.append(g)
  if term or trunc:
    print("episode ended at step", i, "term", term, "trunc", trunc)
    obs, info = env.reset()
print("no-dstb 40-step g: min", round(min(gs), 3), "max", round(max(gs), 3),
      "mean", round(float(np.mean(gs)), 3))

# Strong lateral pull adversary, zero ctrl: g should drop (robot destabilized).
env.reset()
gs2 = []
for i in range(60):
  a = np.zeros(env.action_space.shape, dtype=np.float32)
  a[12] = 1.0  # +Fx max
  a[13] = 1.0  # +Fy max
  obs, g, term, trunc, info = env.step(a)
  gs2.append(g)
  if term or trunc:
    print("dstb episode ended at step", i, "term", term, "trunc", trunc,
          "final g", round(g, 3))
    break
print("dstb g: min", round(min(gs2), 3), "max", round(max(gs2), 3),
      "n", len(gs2))

# Render check.
frame = env.render()
print("render frame", None if frame is None else (np.asarray(frame).shape, np.asarray(frame).dtype))
env.close()
print("PROBE OK")
