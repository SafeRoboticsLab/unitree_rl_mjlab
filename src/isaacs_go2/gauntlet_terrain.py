"""Progressive 'gauntlet' terrain for evaluating how far the crossing policy runs.

A single long track along +x: approach -> [gap_0, platform_0] -> [gap_1,
platform_1] -> ... where gaps GROW and platforms SHRINK with each segment. The
robot starts moving forward at the near edge of gap_0 (patch origin) and the eval
measures how far it gets (which gap it fails at) before falling.
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
class GauntletTerrainCfg(SubTerrainCfg):
  approach_length: float = 2.0
  n_gaps: int = 15
  gap_start: float = 0.15
  gap_growth: float = 0.04     # gap_i = gap_start + i*gap_growth
  plat_start: float = 1.40
  plat_shrink: float = 0.06    # plat_i = max(plat_min, plat_start - i*plat_shrink)
  plat_min: float = 0.50
  gap_depth: float = 0.4  # shallow: a fall can't reach high speed -> no contact-solver blow-up
  final_platform_length: float = 25.0  # long run-out so a success stays up the full episode

  def gap_far_edges(self) -> list[float]:
    """x (relative to origin=near edge of gap_0) of each gap's FAR edge."""
    edges, x = [], 0.0
    for i in range(self.n_gaps):
      gap = self.gap_start + i * self.gap_growth
      plat = max(self.plat_min, self.plat_start - i * self.plat_shrink)
      x += gap
      edges.append(x)
      x += plat
    return edges

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    tw = self.size[1]

    # Approach platform [0, approach_length]; origin at its far edge = gap_0 near edge.
    _add_box(body, geoms, pos=(self.approach_length / 2, tw / 2, 0.0),
             size=(self.approach_length / 2, tw / 2, 0.01))
    x = self.approach_length
    for i in range(self.n_gaps):
      gap = self.gap_start + i * self.gap_growth
      plat = max(self.plat_min, self.plat_start - i * self.plat_shrink)
      # gap pit floor
      _add_box(body, geoms, pos=(x + gap / 2, tw / 2, -self.gap_depth),
               size=(gap / 2, tw / 2, 0.02), rgba=(0.15, 0.05, 0.05, 1.0))
      x += gap
      # (shrinking) landing platform
      shade = 0.35 + 0.03 * i
      _add_box(body, geoms, pos=(x + plat / 2, tw / 2, 0.0),
               size=(plat / 2, tw / 2, 0.01), rgba=(shade, shade, shade, 1.0))
      x += plat

    # Long final run-out platform (so a full clear stays upright the whole episode).
    _add_box(body, geoms, pos=(x + self.final_platform_length / 2, tw / 2, 0.0),
             size=(self.final_platform_length / 2, tw / 2, 0.01))

    origin = np.array([self.approach_length, tw / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geoms)


GAUNTLET_CFG = GauntletTerrainCfg(proportion=1.0)

GAUNTLET_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=(48.0, 2.0),
  border_width=5.0,
  num_rows=8,
  num_cols=8,
  color_scheme="none",
  sub_terrains={"gauntlet": GAUNTLET_CFG},
)
