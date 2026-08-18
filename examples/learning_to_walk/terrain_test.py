import numpy as np
import pytest

from terrain import BOX_CELL
from terrain import BOX_CELLS
from terrain import BOX_HEIGHT
from terrain import HILL_CELL
from terrain import HILL_HEIGHT
from terrain import HILL_OCTAVES
from terrain import HILL_WAVELENGTH
from terrain import BOX_ROWS
from terrain import BOX_WIDTH
from terrain import PLATFORM_RADIUS
from terrain import ROUGH_HEIGHT
from terrain import ROUGH_STEP
from terrain import SLOPE
from terrain import SLOPE_CELL
from terrain import SLOPE_RADIUS
from terrain import TERRAIN_NAMES
from terrain import build_boxes
from terrain import build_hills
from terrain import build_inverted_pyramid
from terrain import build_pyramid
from terrain import build_rough
from terrain import build_terrain
from terrain import describe_terrain


def compute_heights(terrain):
    return terrain.elevations * terrain.peak


def test_every_terrain_spans_the_full_normalized_elevation_range():
    # MuJoCo rescales a heightfield to [0, 1], so an elevation that fell
    # short here would come out taller than the peak asked for.
    for name in TERRAIN_NAMES[1:]:
        terrain = build_terrain(name, 0, 1.0)
        assert terrain.elevations.min() == pytest.approx(0.0)
        assert terrain.elevations.max() == pytest.approx(1.0)


def test_rough_only_uses_the_trained_one_centimetre_height_steps():
    heights = compute_heights(build_terrain("rough", 0, 1.0))
    steps = np.unique(np.round(heights, 6))
    assert np.allclose(steps, np.arange(0.0, ROUGH_HEIGHT + 1e-9, ROUGH_STEP))


def test_rough_difficulty_scales_the_tallest_rock():
    assert build_terrain("rough", 0, 0.5).peak == pytest.approx(0.03)


def test_rough_redraws_its_layout_for_a_new_seed():
    first = build_terrain("rough", 0, 1.0).elevations
    assert not np.allclose(first, build_terrain("rough", 1, 1.0).elevations)


def test_slope_holds_the_trained_grade_outside_the_platform():
    heights = compute_heights(build_terrain("slope", 0, 1.0))
    center = heights.shape[0] // 2
    outside = round((PLATFORM_RADIUS + 1.0) / SLOPE_CELL)
    grades = np.diff(heights[center, center + outside:-1]) / SLOPE_CELL
    assert np.allclose(grades, -SLOPE)


def test_inverted_slope_climbs_where_the_slope_descends():
    down = compute_heights(build_terrain("slope", 0, 1.0))
    up = compute_heights(build_terrain("inverted_slope", 0, 1.0))
    assert np.allclose(down + up, down.max())


def test_slope_is_flat_across_the_platform_the_robot_spawns_on():
    heights = compute_heights(build_terrain("slope", 0, 1.0))
    center = heights.shape[0] // 2
    inside = round(PLATFORM_RADIUS / SLOPE_CELL) - 1
    platform = heights[center, center - inside:center + inside]
    assert np.allclose(platform, heights.max())


def test_slope_rises_over_the_patch_at_the_trained_grade():
    terrain = build_terrain("slope", 0, 1.0)
    span = SLOPE_RADIUS - PLATFORM_RADIUS
    assert terrain.peak == pytest.approx(SLOPE * span)


def test_boxes_are_flat_topped_squares_of_the_trained_width():
    terrain = build_terrain("boxes", 0, 1.0)
    box = terrain.elevations[:BOX_CELLS, :BOX_CELLS]
    assert np.allclose(box, box[0, 0])
    assert BOX_CELLS * BOX_CELL == pytest.approx(BOX_WIDTH)
    assert terrain.radius * 2 == pytest.approx(BOX_ROWS * BOX_WIDTH - BOX_CELL)
    neighbour = terrain.elevations[0, BOX_CELLS]
    assert not np.isclose(neighbour, box[0, 0])


def test_boxes_never_step_higher_than_the_trained_five_centimetres():
    assert compute_heights(build_terrain("boxes", 0, 1.0)).max() <= BOX_HEIGHT


def test_rough_keeps_its_height_step_beyond_the_trained_range():
    heights = compute_heights(build_terrain("rough", 0, 2.0))
    steps = np.unique(np.round(heights, 6))
    assert steps.max() == pytest.approx(2 * ROUGH_HEIGHT)
    assert np.allclose(np.diff(steps), ROUGH_STEP)


def test_slope_keeps_scaling_its_grade_beyond_the_trained_range():
    terrain = build_terrain("slope", 0, 3.0)
    span = SLOPE_RADIUS - PLATFORM_RADIUS
    assert terrain.peak / span == pytest.approx(3 * SLOPE)


def test_every_terrain_stays_finite_at_a_difficulty_near_zero():
    for name in TERRAIN_NAMES[1:]:
        terrain = build_terrain(name, 0, 0.01)
        assert np.isfinite(terrain.elevations).all()


def test_description_flags_a_difficulty_past_the_trained_range():
    assert "OUT OF DISTRIBUTION" in describe_terrain("rough", 1.01)
    assert "OUT OF DISTRIBUTION" not in describe_terrain("rough", 1.0)
    assert "OUT OF DISTRIBUTION" not in describe_terrain("flat", 9.0)


def test_description_states_the_height_the_terrain_is_built_to():
    assert "12.0 cm" in describe_terrain("rough", 2.0)
    assert "10.0 cm" in describe_terrain("boxes", 2.0)
    assert "40 percent" in describe_terrain("inverted_slope", 2.0)


def compute_correlation_length(terrain):
    heights = compute_heights(terrain)
    row = heights[heights.shape[0] // 2]
    correlation = np.correlate(row - row.mean(), row - row.mean(), "full")
    correlation = correlation[len(row) - 1:] / correlation[len(row) - 1]
    cell = 2 * terrain.radius / (heights.shape[0] - 1)
    return np.argmax(correlation < 1 / np.e) * cell


def compute_largest_step(terrain):
    return np.abs(np.diff(compute_heights(terrain), axis=1)).max()


def test_hills_correlate_over_metres_where_rough_correlates_over_one_cell():
    assert compute_correlation_length(build_terrain("rough", 0, 1.0)) <= 0.2
    hills = compute_correlation_length(build_terrain("hills", 0, 1.0))
    assert hills >= HILL_WAVELENGTH / 4


def test_hills_never_step_the_way_uncorrelated_rocks_do():
    rough = build_terrain("rough", 0, 1.0)
    hills = build_terrain("hills", 0, 1.0)
    assert compute_largest_step(rough) == pytest.approx(rough.peak)
    assert compute_largest_step(hills) < 0.2 * hills.peak


def test_hills_hold_the_same_relief_as_rough_so_only_shape_differs():
    assert HILL_HEIGHT == pytest.approx(ROUGH_HEIGHT)
    hills = build_terrain("hills", 0, 1.0)
    assert np.ptp(compute_heights(hills)) == pytest.approx(HILL_HEIGHT)


def test_hills_relief_scales_with_difficulty():
    hills = build_terrain("hills", 0, 3.0)
    relief = np.ptp(compute_heights(hills))
    assert relief == pytest.approx(3 * HILL_HEIGHT)


def test_hills_redraw_their_layout_for_a_new_seed():
    first = build_terrain("hills", 0, 1.0).elevations
    assert not np.allclose(first, build_terrain("hills", 1, 1.0).elevations)


def test_hills_resolve_their_finest_octave_on_the_cell_grid():
    finest = HILL_WAVELENGTH / 2 ** (HILL_OCTAVES - 1)
    assert finest >= 2 * HILL_CELL


def test_hills_are_flagged_out_of_distribution_at_every_difficulty():
    for difficulty in [0.1, 0.5, 1.0, 5.0]:
        assert "OUT OF DISTRIBUTION" in describe_terrain("hills", difficulty)


def test_flat_has_no_heightfield_and_is_not_silently_another_terrain():
    with pytest.raises(ValueError):
        build_terrain("flat", 0, 1.0)


def test_every_name_dispatches_to_the_builder_it_names():
    expected = [("rough", build_rough(0, 1.0)),
                ("slope", build_pyramid(1.0)),
                ("inverted_slope", build_inverted_pyramid(1.0)),
                ("boxes", build_boxes(0, 1.0)),
                ("hills", build_hills(0, 1.0))]
    for name, terrain in expected:
        assert np.allclose(build_terrain(name, 0, 1.0).elevations,
                           terrain.elevations)
