import pytest
import jax.numpy as jp
from unittest.mock import patch

from paz.graphics.renderer import _render
from paz.graphics.types import Shape, Material, Pattern
from paz.graphics import constants as const
from paz import SE3


@pytest.fixture
def dummy_image_shape():
    return 100, 150  # H, W


@pytest.fixture
def dummy_world_to_camera():
    return SE3.translation([0.0, 0.0, -5.0])  # A simple camera transform


@pytest.fixture
def dummy_rays():
    # Simulate rays for a 100x150 image, so 15000 rays
    num_rays = 100 * 150
    origins = jp.zeros((num_rays, 3))
    directions = jp.array([0.0, 0.0, 1.0]) * jp.ones(
        (num_rays, 1)
    )  # All rays pointing forward
    return origins, directions


@pytest.fixture
def dummy_shapes():
    """Provides a list of dummy shapes for the scene."""
    material = Material()
    pattern = Pattern(image=jp.zeros((10, 10, 3)))

    shape1 = Shape(
        transform=jp.eye(4),
        type=const.SPHERE,
        material=material,
        pattern=pattern,
    )
    shape2 = Shape(
        transform=SE3.translation([1.0, 0.0, 0.0]),
        type=const.CUBE,
        material=material,
        pattern=pattern,
    )
    return [shape1, shape2]


@pytest.fixture
def dummy_lights():
    # A simple dummy light, adjust as per your Light type
    class DummyLight:
        pass

    return [DummyLight()]


@pytest.fixture
def dummy_mask():
    """Provides a dummy mask for the image shape."""
    return jp.ones((100, 150), dtype=jp.bool_)  # All pixels initially active


# --- Mocks for dependencies ---


@pytest.fixture
def mock_render_shapes():
    """Mocks the render_shapes function."""
    with patch("paz.graphics.renderer.render_shapes") as mock:
        yield mock


@pytest.fixture
def mock_postprocess():
    """Mocks the postprocess function."""
    with patch("paz.graphics.renderer.postprocess") as mock:
        yield mock


# --- Tests for _render function ---


def test_render_calls_render_shapes_with_correct_args(
    dummy_image_shape,
    dummy_world_to_camera,
    dummy_rays,
    dummy_shapes,
    dummy_lights,
    dummy_mask,
    mock_render_shapes,
    mock_postprocess,
):
    """Verifies _render calls render_shapes with scene elements."""
    # Set up mock to return some plausible data
    num_shapes = len(dummy_shapes)
    num_rays = dummy_rays[0].shape[0]
    mock_render_shapes.return_value = (
        jp.zeros((num_shapes, num_rays), dtype=jp.bool_),  # hit_masks
        jp.zeros((num_shapes, num_rays), dtype=jp.float32),  # depths
        jp.zeros((num_shapes, num_rays, 3), dtype=jp.float32),  # colors
    )

    # Call the function under test
    _render(
        dummy_image_shape,
        dummy_world_to_camera,
        dummy_rays,
        dummy_shapes,
        dummy_lights,
        dummy_mask,
    )

    # Assert render_shapes was called once with the expected arguments
    mock_render_shapes.assert_called_once_with(
        dummy_shapes, dummy_lights, dummy_rays
    )


def test_render_calls_postprocess_with_correct_args(
    dummy_image_shape,
    dummy_world_to_camera,
    dummy_rays,
    dummy_shapes,
    dummy_lights,
    dummy_mask,
    mock_render_shapes,
    mock_postprocess,
):
    """Verifies _render calls postprocess with processed rendering data."""
    # Prepare dummy return values from render_shapes
    num_shapes = len(dummy_shapes)
    num_rays = dummy_rays[0].shape[0]
    mock_hit_masks = jp.array([[True, False], [False, True]])  # Example hits
    mock_depths = jp.array([[1.0, 1e6], [1e6, 2.0]])  # Example depths
    mock_colors = jp.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )  # Example colors

    mock_render_shapes.return_value = (mock_hit_masks, mock_depths, mock_colors)

    # Mock postprocess to return a dummy image and depth map
    mock_postprocess.return_value = (
        jp.zeros(
            (dummy_image_shape[0], dummy_image_shape[1], 3)
        ),  # Dummy image
        jp.zeros(
            (dummy_image_shape[0], dummy_image_shape[1])
        ),  # Dummy depth map
    )

    # Call the function under test
    _render(
        dummy_image_shape,
        dummy_world_to_camera,
        dummy_rays,
        dummy_shapes,
        dummy_lights,
        dummy_mask,
    )

    # Assert postprocess was called once
    mock_postprocess.assert_called_once()

    # Extract arguments passed to postprocess
    args, kwargs = mock_postprocess.call_args

    # Verify the first few arguments
    assert jp.array_equal(
        args[0], mock_hit_masks
    )  # hit_masks from render_shapes
    assert jp.array_equal(args[1], mock_depths)  # depths from render_shapes
    assert jp.array_equal(args[2], mock_colors)  # colors from render_shapes
    assert args[3] is dummy_world_to_camera  # world_to_camera
    assert args[4] is dummy_rays[0]  # ray_origins
    assert args[5] is dummy_rays[1]  # ray_directions
    assert args[6] == dummy_image_shape[0]  # H
    assert args[7] == dummy_image_shape[1]  # W


def test_render_masking_logic_filters_hit_masks(
    dummy_image_shape,
    dummy_world_to_camera,
    dummy_rays,
    dummy_shapes,
    dummy_lights,
    mock_render_shapes,
    mock_postprocess,
):
    """Verifies that the mask correctly filters hit_masks."""
    num_shapes = len(dummy_shapes)
    num_rays = dummy_rays[0].shape[0]

    # Simulate some hits from render_shapes
    initial_hit_masks = jp.array([[True, True, False], [False, True, True]])
    initial_depths = jp.array([[1.0, 2.0, 1e6], [1e6, 3.0, 4.0]])
    initial_colors = jp.zeros((num_shapes, num_rays, 3))
    mock_render_shapes.return_value = (
        initial_hit_masks,
        initial_depths,
        initial_colors,
    )

    # Create a mask that disables some rays
    # Mask is (H, W), here converting to (num_rays,) where it's False for second ray
    dummy_mask_filtered = jp.ones(dummy_image_shape, dtype=jp.bool_).flatten()
    dummy_mask_filtered = dummy_mask_filtered.at[1].set(
        False
    )  # Turn off the second ray

    # Mock postprocess to return dummy values
    mock_postprocess.return_value = (
        jp.zeros((dummy_image_shape[0], dummy_image_shape[1], 3)),
        jp.zeros((dummy_image_shape[0], dummy_image_shape[1])),
    )

    _render(
        dummy_image_shape,
        dummy_world_to_camera,
        dummy_rays,
        dummy_shapes,
        dummy_lights,
        dummy_mask_filtered,
    )

    args, _ = mock_postprocess.call_args
    filtered_hit_masks = args[0]  # The hit_masks passed to postprocess

    # Expected: The second column (ray index 1) should now be False for all shapes
    expected_hit_masks = jp.array([[True, False, False], [False, False, True]])
    assert jp.array_equal(filtered_hit_masks, expected_hit_masks)


def test_render_masking_logic_filters_depths(
    dummy_image_shape,
    dummy_world_to_camera,
    dummy_rays,
    dummy_shapes,
    dummy_lights,
    mock_render_shapes,
    mock_postprocess,
):
    """Verifies that the mask correctly filters depths."""
    num_shapes = len(dummy_shapes)
    num_rays = dummy_rays[0].shape[0]

    initial_hit_masks = jp.array([[True, True, False], [False, True, True]])
    initial_depths = jp.array([[1.0, 2.0, 1e6], [1e6, 3.0, 4.0]])
    initial_colors = jp.zeros((num_shapes, num_rays, 3))
    mock_render_shapes.return_value = (
        initial_hit_masks,
        initial_depths,
        initial_colors,
    )

    # Create a mask that disables some rays
    dummy_mask_filtered = jp.ones(dummy_image_shape, dtype=jp.bool_).flatten()
    dummy_mask_filtered = dummy_mask_filtered.at[1].set(
        False
    )  # Turn off the second ray

    mock_postprocess.return_value = (
        jp.zeros((dummy_image_shape[0], dummy_image_shape[1], 3)),
        jp.zeros((dummy_image_shape[0], dummy_image_shape[1])),
    )

    _render(
        dummy_image_shape,
        dummy_world_to_camera,
        dummy_rays,
        dummy_shapes,
        dummy_lights,
        dummy_mask_filtered,
    )

    args, _ = mock_postprocess.call_args
    filtered_depths = args[1]  # The depths passed to postprocess

    # Expected: Depths for the second column (ray index 1) should be 1e6 (FARAWAY)
    expected_depths = jp.array([[1.0, 1e6, 1e6], [1e6, 1e6, 4.0]])
    assert jp.array_equal(filtered_depths, expected_depths)


def test_render_returns_expected_types(
    dummy_image_shape,
    dummy_world_to_camera,
    dummy_rays,
    dummy_shapes,
    dummy_lights,
    dummy_mask,
    mock_render_shapes,
    mock_postprocess,
):
    """Verifies that _render returns an image and a depth map."""
    num_shapes = len(dummy_shapes)
    num_rays = dummy_rays[0].shape[0]
    mock_render_shapes.return_value = (
        jp.zeros((num_shapes, num_rays), dtype=jp.bool_),
        jp.zeros((num_shapes, num_rays), dtype=jp.float32),
        jp.zeros((num_shapes, num_rays, 3), dtype=jp.float32),
    )

    # Define expected return types from postprocess
    expected_image = jp.zeros((dummy_image_shape[0], dummy_image_shape[1], 3))
    expected_depth_map = jp.zeros((dummy_image_shape[0], dummy_image_shape[1]))
    mock_postprocess.return_value = (expected_image, expected_depth_map)

    image, depth_map = _render(
        dummy_image_shape,
        dummy_world_to_camera,
        dummy_rays,
        dummy_shapes,
        dummy_lights,
        dummy_mask,
    )

    assert isinstance(image, jp.ndarray)
    assert isinstance(depth_map, jp.ndarray)
    assert image.shape == (dummy_image_shape[0], dummy_image_shape[1], 3)
    assert depth_map.shape == dummy_image_shape
    assert jp.array_equal(image, expected_image)
    assert jp.array_equal(depth_map, expected_depth_map)
