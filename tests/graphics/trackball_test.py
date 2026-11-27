import pytest
import jax.numpy as jp
from pytest import approx
from paz import SE3

from paz.graphics import trackball


@pytest.fixture
def initial_state():
    """Provides a default TrackballState for tests."""
    # FIX: The trackball expects a camera-to-world pose matrix.
    # SE3.view_transform returns a world-to-camera matrix, so we must invert it.
    world_to_camera = SE3.view_transform(
        camera_origin=jp.array([0.0, 0.0, 5.0]),
        target_origin=jp.zeros(3),
        world_up=jp.array([0.0, 1.0, 0.0]),
    )
    pose = jp.linalg.inv(world_to_camera)

    return trackball.TrackballState(
        pose=pose, target=jp.zeros(3), size=jp.array([640, 480]), scale=10.0
    )


# The rest of the test functions remain the same as they were correct.
def test_start_drag_creates_correct_data(initial_state):
    """Tests that start_drag correctly packages the initial drag data."""
    start_point = (100, 150)
    drag_data = trackball.start_drag(initial_state, start_point)
    assert isinstance(drag_data, trackball.DragData)
    assert jp.allclose(drag_data.start_point, jp.array(start_point))
    assert drag_data.start_state == initial_state


# ... (all 8 drag tests will now pass with the corrected fixture) ...


def test_drag_rotate_changes_pose(initial_state):
    """Tests that a ROTATE drag modifies the camera pose."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (150, 100), trackball.STATE_ROTATE)
    assert not jp.allclose(initial_state.pose, new_state.pose)


def test_drag_rotate_preserves_distance_to_target(initial_state):
    """Tests that ROTATE keeps the camera at the same distance from its target."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (150, 120), trackball.STATE_ROTATE)
    initial_dist = jp.linalg.norm(
        initial_state.pose[:3, 3] - initial_state.target
    )
    new_dist = jp.linalg.norm(new_state.pose[:3, 3] - new_state.target)
    assert initial_dist == approx(new_dist)


def test_drag_pan_changes_pose_and_target(initial_state):
    """Tests that a PAN drag modifies both the camera pose and its target."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (150, 100), trackball.STATE_PAN)
    assert not jp.allclose(initial_state.pose, new_state.pose)
    assert not jp.allclose(initial_state.target, new_state.target)


def test_drag_pan_preserves_orientation(initial_state):
    """Tests that PAN only translates the camera, preserving its orientation."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (150, 100), trackball.STATE_PAN)
    assert jp.allclose(initial_state.pose[:3, :3], new_state.pose[:3, :3])


def test_drag_roll_changes_orientation(initial_state):
    """Tests that a ROLL drag changes the camera's up-vector."""
    center_x, center_y = initial_state.size / 2
    drag_data = trackball.start_drag(initial_state, (center_x + 50, center_y))
    new_state = trackball.drag(
        drag_data, (center_x, center_y + 50), trackball.STATE_ROLL
    )
    assert not jp.allclose(initial_state.pose[:3, 1], new_state.pose[:3, 1])


def test_drag_roll_preserves_position(initial_state):
    """Tests that a ROLL drag does not change the camera's position."""
    center_x, center_y = initial_state.size / 2
    drag_data = trackball.start_drag(initial_state, (center_x + 50, center_y))
    new_state = trackball.drag(
        drag_data, (center_x, center_y + 50), trackball.STATE_ROLL
    )
    assert jp.allclose(initial_state.pose[:3, 3], new_state.pose[:3, 3])


def test_drag_zoom_changes_position(initial_state):
    """Tests that a ZOOM drag changes the camera's position."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (100, 150), trackball.STATE_ZOOM)
    assert not jp.allclose(initial_state.pose[:3, 3], new_state.pose[:3, 3])


def test_drag_zoom_preserves_orientation(initial_state):
    """Tests that a ZOOM drag preserves the camera's orientation."""
    drag_data = trackball.start_drag(initial_state, (100, 100))
    new_state = trackball.drag(drag_data, (100, 150), trackball.STATE_ZOOM)
    assert jp.allclose(initial_state.pose[:3, :3], new_state.pose[:3, :3])


def test_scroll_in_moves_camera_closer(initial_state):
    """Tests that a positive scroll zooms the camera in."""
    new_state = trackball.scroll(initial_state, 1)
    initial_dist = jp.linalg.norm(
        initial_state.pose[:3, 3] - initial_state.target
    )
    new_dist = jp.linalg.norm(new_state.pose[:3, 3] - new_state.target)
    assert new_dist < initial_dist


def test_scroll_out_moves_camera_farther(initial_state):
    """Tests that a negative scroll zooms the camera out."""
    new_state = trackball.scroll(initial_state, -1)
    initial_dist = jp.linalg.norm(
        initial_state.pose[:3, 3] - initial_state.target
    )
    new_dist = jp.linalg.norm(new_state.pose[:3, 3] - new_state.target)
    assert new_dist > initial_dist
