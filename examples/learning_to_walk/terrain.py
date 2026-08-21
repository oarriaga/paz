"""Heightfields for testing the policy on and off its training terrain.

The cell sizes, height ranges and slopes of rough, the slopes and boxes
come from ROBUST_TERRAINS_CFG in the run's params/velocity_env_robust_cfg.py.
IsaacLab lays those out as 8 by 8 metre tiles. The patches here span 20
metres so a run never walks off one, which keeps the local grade and cell
size rather than the tile size.

Hills is the exception and is in no training mix at all. It exists to test
a surface whose relief is spatially correlated, which none of the others
are at any difficulty.
"""

from collections import namedtuple

import numpy as np
from scipy import ndimage

TERRAIN_NAMES = ("flat", "rough", "slope", "inverted_slope", "boxes",
                 "hills")

ROUGH_CELL = 0.1
ROUGH_CELLS = 200
ROUGH_STEP = 0.01
ROUGH_HEIGHT = 0.06
ROUGH_RADIUS = (ROUGH_CELLS - 1) * ROUGH_CELL / 2

SLOPE_CELL = 0.1
SLOPE_CELLS = 200
SLOPE = 0.2
SLOPE_RADIUS = (SLOPE_CELLS - 1) * SLOPE_CELL / 2
PLATFORM_RADIUS = 1.0

BOX_CELL = 0.05
BOX_WIDTH = 0.45
BOX_CELLS = round(BOX_WIDTH / BOX_CELL)
BOX_ROWS = 45
BOX_HEIGHT = 0.05
BOX_CELL_COUNT = BOX_ROWS * BOX_CELLS
BOX_RADIUS = (BOX_CELL_COUNT - 1) * BOX_CELL / 2

# Nothing in the training mix is spatially correlated: every rock and every
# box is drawn independently of its neighbours. These hills sum smooth
# octaves instead, so the ground rolls the way real ground does, and no
# amount of difficulty turns one terrain into the other.
HILL_CELL = 0.1
HILL_CELLS = 200
HILL_RADIUS = (HILL_CELLS - 1) * HILL_CELL / 2
HILL_HEIGHT = 0.06
HILL_WAVELENGTH = 2.0
HILL_OCTAVES = 4
HILL_PERSISTENCE = 0.5

Terrain = namedtuple("Terrain", "elevations peak radius")
Extent = namedtuple("Extent", "summary limit trained")


def build_terrain(name, seed, difficulty):
    if name == "rough":
        terrain = build_rough(seed, difficulty)
    elif name == "slope":
        terrain = build_pyramid(difficulty)
    elif name == "inverted_slope":
        terrain = build_inverted_pyramid(difficulty)
    elif name == "boxes":
        terrain = build_boxes(seed, difficulty)
    elif name == "hills":
        terrain = build_hills(seed, difficulty)
    else:
        raise ValueError(f"{name} has no heightfield")
    return terrain


def describe_terrain(name, difficulty):
    if name == "flat":
        description = "flat, the released ground plane"
    else:
        extent = compute_extent(name, difficulty)
        note = describe_distribution(difficulty, extent)
        header = f"{name} at difficulty {difficulty:.2f}"
        description = f"{header}: {extent.summary}\n  {note}"
    return description


def describe_distribution(difficulty, extent):
    if not extent.trained:
        note = f"OUT OF DISTRIBUTION: {extent.limit}"
    elif difficulty > 1.0:
        ratio = f"{difficulty:.2f}x the {extent.limit}"
        note = f"OUT OF DISTRIBUTION: {ratio} it trained on"
    else:
        note = f"inside the training range, which reached {extent.limit}"
    return note


def compute_extent(name, difficulty):
    if name == "rough":
        extent = compute_rough_extent(difficulty)
    elif name in ("slope", "inverted_slope"):
        extent = compute_slope_extent(difficulty)
    elif name == "boxes":
        extent = compute_boxes_extent(difficulty)
    elif name == "hills":
        extent = compute_hills_extent(difficulty)
    else:
        raise ValueError(f"{name} has no extent")
    return extent


def compute_rough_extent(difficulty):
    peak, step = ROUGH_HEIGHT * difficulty * 100, ROUGH_STEP * 100
    summary = f"rocks up to {peak:.1f} cm in {step:.1f} cm steps"
    return Extent(summary, f"{ROUGH_HEIGHT * 100:.1f} cm rocks", True)


def compute_slope_extent(difficulty):
    summary = f"a {SLOPE * difficulty * 100:.0f} percent grade"
    return Extent(summary, f"{SLOPE * 100:.0f} percent grade", True)


def compute_boxes_extent(difficulty):
    peak, width = BOX_HEIGHT * difficulty * 100, BOX_WIDTH * 100
    summary = f"{width:.0f} cm boxes up to {peak:.1f} cm tall"
    return Extent(summary, f"{BOX_HEIGHT * 100:.1f} cm boxes", True)


def compute_hills_extent(difficulty):
    relief = HILL_HEIGHT * difficulty * 100
    finest = HILL_WAVELENGTH / 2 ** (HILL_OCTAVES - 1)
    summary = (f"rolling ground {relief:.1f} cm deep, from"
               f" {HILL_WAVELENGTH:.1f} m down to {finest:.2f} m")
    novelty = "training never showed it correlated ground"
    return Extent(summary, novelty, False)


def build_rough(seed, difficulty):
    peak = ROUGH_HEIGHT * difficulty
    levels = max(round(peak / ROUGH_STEP), 1)
    shape = (ROUGH_CELLS, ROUGH_CELLS)
    steps = np.random.default_rng(seed).integers(levels + 1, size=shape)
    return Terrain(steps / levels, peak, ROUGH_RADIUS)


def build_pyramid(difficulty):
    ramp = build_ramp()
    return Terrain(1.0 - ramp, compute_rise(difficulty), SLOPE_RADIUS)


def build_inverted_pyramid(difficulty):
    ramp = build_ramp()
    return Terrain(ramp, compute_rise(difficulty), SLOPE_RADIUS)


def build_boxes(seed, difficulty):
    shape = (BOX_ROWS, BOX_ROWS)
    heights = np.random.default_rng(seed).uniform(size=shape)
    block = np.ones((BOX_CELLS, BOX_CELLS))
    peak = BOX_HEIGHT * difficulty
    elevations = np.kron(normalize(heights), block)
    return Terrain(elevations, peak, BOX_RADIUS)


def build_hills(seed, difficulty):
    rng = np.random.default_rng(seed)
    heights = np.zeros((HILL_CELLS, HILL_CELLS))
    for octave in range(HILL_OCTAVES):
        heights = heights + build_octave(rng, octave)
    peak = HILL_HEIGHT * difficulty
    return Terrain(normalize(heights), peak, HILL_RADIUS)


def build_octave(rng, octave):
    wavelength = HILL_WAVELENGTH / 2 ** octave
    cells = round(HILL_CELLS * HILL_CELL / wavelength)
    coarse = rng.normal(size=(cells, cells))
    zoom = HILL_CELLS / cells
    smooth = ndimage.zoom(coarse, zoom, order=3, mode="grid-wrap")
    return smooth * HILL_PERSISTENCE ** octave


def normalize(heights):
    # MuJoCo rescales every heightfield to span [0, 1]. Doing it here keeps
    # these elevations equal to the ones the simulator ends up rendering.
    return (heights - heights.min()) / np.ptp(heights)


def build_ramp():
    axis = np.linspace(-SLOPE_RADIUS, SLOPE_RADIUS, SLOPE_CELLS)
    distance = np.maximum(np.abs(axis[:, np.newaxis]), np.abs(axis))
    span = SLOPE_RADIUS - PLATFORM_RADIUS
    return np.clip((distance - PLATFORM_RADIUS) / span, 0.0, 1.0)


def compute_rise(difficulty):
    return SLOPE * difficulty * (SLOPE_RADIUS - PLATFORM_RADIUS)
