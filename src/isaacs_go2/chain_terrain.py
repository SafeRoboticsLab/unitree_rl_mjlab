"""Chained-gap TRAINING terrain for learning to cross a sequence of gaps.

Unlike the fixed progressive ``gauntlet`` (an eval terrain), this is a
difficulty-curriculum terrain: each patch is a chain of ``n_gaps`` gaps
interleaved with run-up platforms, where the gaps GROW and the platforms SHRINK
as ``difficulty`` rises (0 -> 1).  Within a single patch the gaps also vary in
size (``gap_step``) so the robot faces interleaved widths, not one fixed width.

Paired with ``terrain_levels_parkour`` (promote an env to a harder row when it
traverses >50 % of the patch = chains several gaps), the robot is pulled from
easy (small gaps, long platforms — momentum easy to build) toward hard (wide
gaps, short platforms — must run hard and commit) only as fast as it masters
each level.  Gaps are SHALLOW (``gap_depth`` ~0.4 m) so a missed jump ends in a
clean early termination instead of a deep free-fall that blows up the contact
solver (learned from the gauntlet eval).

Layout along +x (patch origin at the approach start = where the robot spawns)::

    [ approach ] [gap_0][plat_0] [gap_1][plat_1] ... [gap_{n-1}][plat_{n-1}] [ run-out ]
      spawn/run                                                                stays safe
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)


def _add_box(body, geoms, pos, size, rgba=(0.5, 0.5, 0.5, 1.0)):
  geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
  geom.rgba = rgba
  geoms.append(TerrainGeometry(geom=geom, color=rgba))


@dataclass(kw_only=True)
class ChainCrossingTerrainCfg(SubTerrainCfg):
  approach_length: float = 1.5
  n_gaps: int = 6
  # gap_i = (gap_base_min + difficulty*(gap_base_max-gap_base_min)) + i*gap_step
  gap_base_min: float = 0.10   # first gap width at difficulty 0
  gap_base_max: float = 0.45   # first gap width at difficulty 1
  gap_step: float = 0.03       # per-gap growth within a patch (interleaved sizes)
  # plat_i = plat_max - difficulty*(plat_max-plat_min)  (run-up shrinks w/ difficulty)
  plat_max: float = 1.80
  plat_min: float = 1.30
  gap_depth: float = 0.4       # shallow -> clean early termination, no blow-up

  def _sizes(self, difficulty: float) -> tuple[list[float], float]:
    gap_base = self.gap_base_min + difficulty * (self.gap_base_max - self.gap_base_min)
    gaps = [gap_base + i * self.gap_step for i in range(self.n_gaps)]
    plat = self.plat_max - difficulty * (self.plat_max - self.plat_min)
    return gaps, plat

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    tw = self.size[1]
    gaps, plat = self._sizes(float(difficulty))

    # Approach platform [0, approach_length]; origin at its START (spawn point).
    _add_box(body, geoms, pos=(self.approach_length / 2, tw / 2, 0.0),
             size=(self.approach_length / 2, tw / 2, 0.01))
    x = self.approach_length
    for i in range(self.n_gaps):
      gap = gaps[i]
      # shallow gap pit floor (visual + catches a short fall for clean termination)
      _add_box(body, geoms, pos=(x + gap / 2, tw / 2, -self.gap_depth),
               size=(gap / 2, tw / 2, 0.02), rgba=(0.15, 0.05, 0.05, 1.0))
      x += gap
      # run-up / landing platform
      shade = 0.35 + 0.02 * i
      _add_box(body, geoms, pos=(x + plat / 2, tw / 2, 0.0),
               size=(plat / 2, tw / 2, 0.01), rgba=(shade, shade, shade, 1.0))
      x += plat

    # Long run-out platform (robustly safe: a full chain stays upright here).
    run_out = max(0.5, self.size[0] - x)
    _add_box(body, geoms, pos=(x + run_out / 2, tw / 2, 0.0),
             size=(run_out / 2, tw / 2, 0.01))

    origin = np.array([0.0, tw / 2, 0.0])  # spawn at approach start
    return TerrainOutput(origin=origin, geometries=geoms)


CHAIN_CROSSING_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(16.0, 3.0),
  border_width=5.0,
  num_rows=10,     # difficulty levels (terrain_levels_parkour promotes up rows)
  num_cols=10,     # variations per level
  difficulty_range=(0.0, 1.0),
  color_scheme="none",
  sub_terrains={
    "chain": ChainCrossingTerrainCfg(proportion=1.0),
  },
)
