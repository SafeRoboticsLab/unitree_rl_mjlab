"""On-policy ISAACS league: vendored leaderboard + env-slice opponent sampling.

The ``Leaderboard`` class is vendored from
``safety-stable-baselines/safety_sb3/leaderboard.py`` (pure numpy/torch — the
safety_sb3 package is not installed in the mjlab env and its __init__ imports
the whole SB3 stack).  Semantics identical to safe_adaptation_dev: checkpoint
archive of ctrl (max) / dstb (min) actors, score matrix, evict-worst pruning,
softmax-rationality opponent sampling.

The on-policy addition is :meth:`Leaderboard.sample_dstb_slices`: instead of
one opponent per rollout (SAC style), different frozen disturbances are
assigned to different SLICES of the parallel envs — one PPO batch contains
rollouts against the whole opponent population (AlphaStar-league style, per
doc/research_note_onpolicy_league_reach_avoid.md).
"""

from __future__ import annotations

import os

import numpy as np
import torch as th

# Slice opponent codes (aligned with isaacs_runner conventions).
SLICE_ZERO = -3     # no disturbance (fills the board's dummy column)
SLICE_RANDOM = -2   # uniform random direction at full force
SLICE_CURRENT = -1  # the live dstb actor
# >= 0: index into dstb_steps (an archived checkpoint)


class Leaderboard:
  """Vendored from safety_sb3 (see module docstring); board (kc+1, kd+2)."""

  def __init__(self, save_top_k_ctrl: int, save_top_k_dstb: int,
               softmax_rationality: float, model_dir: str, seed: int = 0):
    self.kc = int(save_top_k_ctrl)
    self.kd = int(save_top_k_dstb)
    self.rationality = float(softmax_rationality)
    self.dir = model_dir
    os.makedirs(self.dir, exist_ok=True)
    self.rng = np.random.default_rng(seed)
    self.ctrl_steps: list[int] = []
    self.dstb_steps: list[int] = []
    self.board = np.full((self.kc + 1, self.kd + 2), np.nan, dtype=float)

  def _path(self, kind: str, step: int) -> str:
    return os.path.join(self.dir, f"{kind}_{step}.pt")

  def save_actor(self, actor: th.nn.Module, kind: str, step: int) -> None:
    th.save(actor.state_dict(), self._path(kind, step))

  def load_actor(self, actor: th.nn.Module, kind: str, step: int) -> th.nn.Module:
    actor.load_state_dict(th.load(self._path(kind, step), map_location="cpu"))
    return actor

  def _remove(self, kind: str, step: int) -> None:
    p = self._path(kind, step)
    if os.path.exists(p):
      os.remove(p)

  def set_score(self, ctrl_idx: int, dstb_idx: int, metric: float) -> None:
    self.board[ctrl_idx, dstb_idx] = metric

  def ema_score(self, ctrl_idx: int, dstb_idx: int, metric: float,
                beta: float = 0.1) -> None:
    """EMA update — training-outcome scores arrive noisy every rollout."""
    old = self.board[ctrl_idx, dstb_idx]
    if np.isnan(old):
      self.board[ctrl_idx, dstb_idx] = metric
    else:
      self.board[ctrl_idx, dstb_idx] = (1 - beta) * old + beta * metric

  def prune(self, step: int, ctrl_actor: th.nn.Module,
            dstb_actor: th.nn.Module) -> None:
    if len(self.ctrl_steps) == self.kc:
      ctrl_avg = np.nanmean(self.board, axis=1)
      worst = int(np.argmin(ctrl_avg))
      if worst != self.kc:
        self._remove("ctrl", self.ctrl_steps[worst])
        self.ctrl_steps[worst] = step
        self.board[worst] = self.board[-1]
        self.save_actor(ctrl_actor, "ctrl", step)
    else:
      self.ctrl_steps.append(step)
      self.save_actor(ctrl_actor, "ctrl", step)

    if len(self.dstb_steps) == self.kd:
      dstb_avg = np.nanmean(self.board[:, :-1], axis=0)
      worst = int(np.argmax(dstb_avg))
      if worst != self.kd:
        self._remove("dstb", self.dstb_steps[worst])
        self.dstb_steps[worst] = step
        self.board[:, worst] = self.board[:, -2]
        self.save_actor(dstb_actor, "dstb", step)
    else:
      self.dstb_steps.append(step)
      self.save_actor(dstb_actor, "dstb", step)

  # --- on-policy addition: opponents over env slices ------------------------

  def sample_dstb_slices(self, n_slices: int) -> list[int]:
    """Opponent per env slice for one ctrl-phase rollout.

    Slice 0 = ZERO (keeps no-push behavior trained AND fills the dummy
    column), slice 1 = RANDOM (increment-0 insurance); the rest are softmax-
    sampled over {archived dstbs, CURRENT} with p ∝ exp(-rationality * avg):
    disturbances that hurt the ctrl more are replayed more often.
    """
    out = [SLICE_ZERO, SLICE_RANDOM]
    n_free = max(n_slices - 2, 0)
    n_arch = len(self.dstb_steps)
    if n_arch == 0:
      out += [SLICE_CURRENT] * n_free
      return out[:n_slices]
    cols = list(range(n_arch)) + [self.board.shape[1] - 2]  # archived + CURRENT
    with np.errstate(invalid="ignore"):
      logit = np.nanmean(self.board[:, cols], axis=0)
      fill = float(np.nanmean(self.board)) if np.isfinite(self.board).any() else 0.0
    logit = np.nan_to_num(logit, nan=fill)
    p = np.exp(-self.rationality * logit)
    p = p / p.sum()
    for _ in range(n_free):
      pick = int(self.rng.choice(len(cols), p=p))
      out.append(SLICE_CURRENT if cols[pick] == self.board.shape[1] - 2 else pick)
    return out[:n_slices]
