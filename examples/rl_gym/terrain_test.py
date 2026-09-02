import mujoco
import numpy as np

import terrain
from robots.g1 import G1DoF29

SMALL_COUNTS = terrain.TerrainCounts(flat=1, rough=1, slope=1, inverted_slope=1, boxes=1)  # fmt: skip


def build_small_terrain():
    return terrain.build(0, num_levels=2, border_width=1.0, counts=SMALL_COUNTS)


def test_build_places_one_origin_per_tile_at_its_centre_height():
    small = build_small_terrain()
    assert small.origins.shape == (2, 5, 3)
    tile_cells = terrain.compute_num_cells(small.tile_size)
    border_cells = terrain.compute_num_cells(small.border_width)
    region = terrain.compute_tile_region(1, 2, tile_cells, border_cells)
    tile = small.heights[region]
    assert tile.shape == (tile_cells + 1, tile_cells + 1)
    centre = tile_cells // 2
    assert np.isclose(small.origins[1, 2, 2], tile[centre, centre])


def test_rough_tiles_carry_full_noise_at_every_level():
    small = build_small_terrain()
    tile_cells = terrain.compute_num_cells(small.tile_size)
    border_cells = terrain.compute_num_cells(small.border_width)
    for level in range(2):
        region = terrain.compute_tile_region(level, 1, tile_cells, border_cells)
        assert np.isclose(small.heights[region].max(), 0.06)


def test_heightfield_is_split_into_one_field_per_tile():
    small = build_small_terrain()
    mjspec = G1DoF29().mjspec.copy()
    terrain.add_heightfield(mjspec, small)
    model = mjspec.compile()
    assert model.nhfield == 10
    boxes = np.sum(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    fields = np.sum(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
    assert fields == 10 and boxes >= 4
    tile = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tile_1_2")
    assert np.allclose(model.geom_pos[tile][:2], small.origins[1, 2, :2])
