import argparse
from collections import namedtuple
import math

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mujoco
import numpy as np

from paz.utils import plot

Terrain = namedtuple("Terrain", "elevations, peak, minimum, origins, tile_size, border_width, horizontal_scale")  # fmt: skip
TerrainCounts = namedtuple("TerrainCounts", "flat, rough, slope, inverted_slope, boxes")  # fmt: skip
TERRAIN_COUNTS = TerrainCounts(flat=5, rough=8, slope=2, inverted_slope=2, boxes=4)  # fmt: skip

def build(seed, num_levels=9, tile_size=8.0, border_width=20.0, horizontal_scale=0.1, vertical_scale=0.005, counts=TERRAIN_COUNTS):  # fmt: skip
    terrain_types = build_terrain_types(counts)
    tile_cells = compute_num_cells(tile_size, horizontal_scale)
    border_cells = compute_num_cells(border_width, horizontal_scale)
    shape_args = num_levels, len(terrain_types), tile_cells, border_cells
    shape = compute_terrain_shape(*shape_args)
    heights = np.zeros(shape, "float32")
    origins = np.zeros((num_levels, len(terrain_types), 3), "float32")
    rng = np.random.default_rng(seed)
    for level in range(num_levels):
        # the reference draws each tile's difficulty inside its level band
        difficulty = (level + rng.uniform()) / num_levels
        for column, terrain_type in enumerate(terrain_types):
            tile_args = rng, terrain_type, difficulty, tile_size, horizontal_scale, vertical_scale  # fmt: skip
            tile = build_tile(*tile_args)
            region_args = level, column, tile_cells, border_cells
            heights[compute_tile_region(*region_args)] = tile
            origin_args = tile, level, column, shape, tile_size, border_width, horizontal_scale  # fmt: skip
            origins[level, column] = compute_origin(*origin_args)
    minimum = float(heights.min())
    peak = max(float(heights.max()) - minimum, vertical_scale)
    elevations = (heights - minimum) / peak
    terrain_args = elevations, peak, minimum, origins, tile_size, border_width, horizontal_scale  # fmt: skip
    return Terrain(*terrain_args)


def compute_difficulty(level, num_levels):
    return level / (num_levels - 1)


def build_terrain_types(counts):
    terrain_types = ()
    for terrain_type, count in zip(counts._fields, counts):
        terrain_types = terrain_types + (terrain_type,) * count
    return terrain_types


def compute_terrain_shape(num_levels, num_columns, tile_cells, border_cells):
    rows = compute_axis_cells(num_levels, tile_cells, border_cells)
    columns = compute_axis_cells(num_columns, tile_cells, border_cells)
    return rows, columns


def compute_axis_cells(num_tiles, tile_cells, border_cells):
    return num_tiles * tile_cells + 1 + 2 * border_cells


def compute_num_cells(width, horizontal_scale=0.1):
    return round(width / horizontal_scale)


def build_tile(rng, terrain_type, difficulty, tile_size=8.0, horizontal_scale=0.1, vertical_scale=0.005):  # fmt: skip
    tile_cells = compute_num_cells(tile_size, horizontal_scale)
    if terrain_type == "flat":
        tile = np.zeros((tile_cells + 1, tile_cells + 1))
    elif terrain_type == "rough":
        tile = build_rough(rng, tile_cells)
    elif terrain_type == "slope":
        tile = build_pyramid(difficulty, tile_size, tile_cells)
    elif terrain_type == "inverted_slope":
        tile = build_inverted_pyramid(difficulty, tile_size, tile_cells)
    else:
        tile = build_boxes(rng, difficulty, tile_cells, horizontal_scale)
    return quantize(tile, vertical_scale).astype("float32")


def build_rough(rng, tile_cells=80):
    # the reference ignores difficulty for this sub-terrain: every level
    # carries the full 6 cm noise
    shape = tile_cells + 1, tile_cells + 1
    return rng.integers(7, size=shape) * 0.01


def build_pyramid(difficulty, tile_size=8.0, tile_cells=80):
    ramp = 1.0 - build_ramp(tile_size, tile_cells)
    return ramp * 0.2 * difficulty * (tile_size / 2 - 1.0)


def build_inverted_pyramid(difficulty, tile_size=8.0, tile_cells=80):
    ramp = build_ramp(tile_size, tile_cells)
    return ramp * 0.2 * difficulty * (tile_size / 2 - 1.0)


def build_ramp(tile_size=8.0, tile_cells=80):
    axis = np.linspace(-tile_size / 2, tile_size / 2, tile_cells + 1)
    distance = np.maximum(np.abs(axis[:, None]), np.abs(axis[None, :]))
    return np.clip((distance - 1.0) / (tile_size / 2 - 1.0), 0.0, 1.0)


def build_boxes(rng, difficulty, tile_cells=80, horizontal_scale=0.1):
    box_cells = compute_num_cells(0.45, horizontal_scale)
    num_boxes = math.ceil((tile_cells + 1) / box_cells)
    # the reference draws box tops below the platform as well as above
    bound = 0.05 * difficulty
    heights = rng.uniform(-bound, bound, (num_boxes, num_boxes))
    tile = np.repeat(np.repeat(heights, box_cells, axis=0), box_cells, axis=1)
    tile = tile[: tile_cells + 1, : tile_cells + 1]
    center = tile_cells // 2
    platform_cells = compute_num_cells(1.0, horizontal_scale)
    region = slice(center - platform_cells, center + platform_cells + 1)
    tile[region, region] = 0.0
    return tile


def quantize(heights, vertical_scale=0.005):
    return np.round(heights / vertical_scale) * vertical_scale


def compute_tile_region(level, column, tile_cells=80, border_cells=200):
    row = border_cells + level * tile_cells
    column_start = border_cells + column * tile_cells
    rows = slice(row, row + tile_cells + 1)
    columns = slice(column_start, column_start + tile_cells + 1)
    return rows, columns


def compute_origin(tile, level, column, terrain_shape, tile_size=8.0, border_width=20.0, horizontal_scale=0.1):  # fmt: skip
    width_x = (terrain_shape[1] - 1) * horizontal_scale
    width_y = (terrain_shape[0] - 1) * horizontal_scale
    x = -width_x / 2 + border_width + (column + 0.5) * tile_size
    y = -width_y / 2 + border_width + (level + 0.5) * tile_size
    center = compute_num_cells(tile_size, horizontal_scale) // 2
    z = tile[center, center]
    return x, y, z


def add_heightfield(model_spec, terrain):
    rows, columns = terrain.elevations.shape
    size_x = (columns - 1) * terrain.horizontal_scale / 2
    size_y = (rows - 1) * terrain.horizontal_scale / 2
    size = [size_x, size_y, terrain.peak, 0.1]
    field = model_spec.add_hfield(name="terrain_heightfield", size=size)
    field.nrow, field.ncol = rows, columns
    field.userdata = terrain.elevations.ravel()
    floor = model_spec.geom("floor")
    floor.contype = 0
    floor.conaffinity = 0
    # hide the coplanar floor plane, which z-fights with flat tiles
    floor.group = 4
    geom = model_spec.worldbody.add_geom()
    geom.name = "terrain"
    geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
    geom.hfieldname = "terrain_heightfield"
    geom.pos[2] = terrain.minimum
    geom.material = add_terrain_material(model_spec)


def add_terrain_material(model_spec, cells_per_meter=2):
    # a checker texture so translation is visible in rendered videos
    texture = model_spec.add_texture(name="terrain_checker")
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    texture.rgb1, texture.rgb2 = [0.6, 0.6, 0.6], [0.45, 0.5, 0.45]
    texture.width = texture.height = 512
    material = model_spec.add_material(name="terrain_material")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "terrain_checker"
    material.texrepeat = [cells_per_meter, cells_per_meter]
    material.texuniform = True
    return material.name


def draw(terrain, counts=TERRAIN_COUNTS):
    heights = terrain.elevations * terrain.peak + terrain.minimum
    crop_args = heights, terrain.horizontal_scale, terrain.border_width
    tiles = crop_border(*crop_args)
    num_levels, num_columns = terrain.origins.shape[:2]
    figure, axis = plot.subplots(figsize=compute_figsize(tiles.shape))
    extent = -0.5, num_columns - 0.5, -0.5, num_levels - 0.5
    kwargs = {"cmap": build_colormap(), "origin": "lower", "extent": extent, "aspect": "equal"}  # fmt: skip
    image = axis.imshow(tiles, **kwargs)
    draw_tile_grid(axis, num_levels, num_columns)
    label_columns(axis, build_terrain_types(counts))
    label_levels(axis, num_levels)
    add_colorbar(axis, image)
    plot.set_labels(axis, x="terrain type", y="difficulty")
    return figure


def compute_figsize(shape, width=15.0, margin=1.6):
    return width, width * shape[0] / shape[1] + margin


def crop_border(heights, horizontal_scale=0.1, border_width=20.0):
    cells = compute_num_cells(border_width, horizontal_scale)
    return heights[cells:-cells, cells:-cells]


def build_colormap(name="YlGn", low=0.25, num_colors=256, exponent=0.3):
    fractions = np.linspace(0.0, 0.75, num_colors) ** exponent
    samples = low + (1.0 - low) * fractions
    return colors.ListedColormap(plt.colormaps[name](samples))


def draw_tile_grid(axis, num_levels, num_columns, color="white", width=1.5):
    style = {"color": color, "linewidth": width}
    for column in range(num_columns + 1):
        axis.axvline(column - 0.5, **style)
    for level in range(num_levels + 1):
        axis.axhline(level - 0.5, **style)
    axis.tick_params(which="both", bottom=False, left=False)
    plot.hide_spines(axis, "all")


def label_columns(axis, terrain_types):
    centers = {}
    for column, terrain_type in enumerate(terrain_types):
        centers.setdefault(terrain_type, []).append(column)
    names = list(centers)
    axis.set_xticks([np.mean(centers[name]) for name in names])
    axis.set_xticklabels([name.replace("_", " ") for name in names])


def label_levels(axis, num_levels):
    labels = []
    for level in range(num_levels):
        labels.append(f"{compute_difficulty(level, num_levels):.2f}")
    axis.set_yticks(range(num_levels))
    axis.set_yticklabels(labels)


def add_colorbar(axis, image, label="height [m]", size="2%", pad=0.05):
    cax = make_axes_locatable(axis).append_axes("right", size=size, pad=pad)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel(label, rotation=-90, va="bottom", labelpad=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=9)
    parser.add_argument("--tile_size", type=float, default=8.0)
    parser.add_argument("--border_width", type=float, default=20.0)
    parser.add_argument("--horizontal_scale", type=float, default=0.1)
    parser.add_argument("--vertical_scale", type=float, default=0.005)
    parser.add_argument("--filepath", type=str, default="terrain.pdf")
    args = parser.parse_args()
    terrain_args = args.seed, args.num_levels, args.tile_size, args.border_width, args.horizontal_scale, args.vertical_scale  # fmt: skip
    terrain = build(*terrain_args)
    plot.configure(fontsize=11)
    figure = draw(terrain)
    plot.show()
    plot.save(figure, args.filepath)
