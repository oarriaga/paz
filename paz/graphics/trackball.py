import jax
import jax.numpy as jp
from paz import SE3
from paz.backend.lie import SO3
from collections import namedtuple

STATE_ROTATE = 0
STATE_PAN = 1
STATE_ROLL = 2
STATE_ZOOM = 3

TrackballState = namedtuple(
    "TrackballState", ["pose", "target", "size", "scale"]
)
DragData = namedtuple("DragData", ["start_point", "start_state"])


def _rotation_around_point(angle, axis, point):
    """Helper to create a 4x4 matrix for rotation around a point and axis."""
    # FIX: Convert the so(3) vector (axis * angle) to a 3x3 so(3) matrix
    # before passing it to the matrix exponential SO3.exp().
    so3_vector = axis * angle
    so3_matrix = SO3.hat(so3_vector)
    rotation_3x3 = SO3.exp(so3_matrix)

    transform_at_origin = SE3.to_affine_matrix(rotation_3x3, jp.zeros(3))
    translate_to_point = SE3.translation(point)
    translate_from_origin = SE3.translation(-point)
    return translate_to_point @ transform_at_origin @ translate_from_origin


def _apply_rotation(start_state, start_point, current_point):
    """Computes a new state by rotating the camera."""
    dx, dy = current_point - start_point
    mindim = 0.3 * jp.min(start_state.size)
    x_axis, y_axis = start_state.pose[:3, 0], start_state.pose[:3, 1]
    x_angle, y_angle = -dx / mindim, dy / mindim

    x_rot_mat = _rotation_around_point(x_angle, y_axis, start_state.target)
    y_rot_mat = _rotation_around_point(y_angle, x_axis, start_state.target)

    new_pose = y_rot_mat @ x_rot_mat @ start_state.pose
    return start_state._replace(pose=new_pose)


def _apply_pan(start_state, start_point, current_point):
    """Computes a new state by panning the camera."""
    dx, dy = current_point - start_point
    mindim = jp.min(start_state.size)
    x_axis, y_axis = start_state.pose[:3, 0], start_state.pose[:3, 1]
    dx = dx / (5.0 * mindim) * start_state.scale
    dy = -dy / (5.0 * mindim) * start_state.scale

    translation = dx * x_axis + dy * y_axis
    pan_transform = SE3.translation(translation)

    new_pose = pan_transform @ start_state.pose
    new_target = start_state.target + translation
    return start_state._replace(pose=new_pose, target=new_target)


def _apply_roll(start_state, start_point, current_point):
    """Computes a new state by rolling the camera."""
    center = start_state.size / 2.0
    v_init = start_point - center
    v_curr = current_point - center
    v_init /= jp.linalg.norm(v_init) + 1e-8
    v_curr /= jp.linalg.norm(v_curr) + 1e-8

    theta = jp.arctan2(v_init[1], v_init[0]) - jp.arctan2(v_curr[1], v_curr[0])
    z_axis = start_state.pose[:3, 2]
    roll_transform = _rotation_around_point(theta, z_axis, start_state.target)

    new_pose = roll_transform @ start_state.pose
    return start_state._replace(pose=new_pose)


def _apply_zoom(start_state, start_point, current_point):
    """Computes a new state by zooming the camera."""
    _, dy = current_point - start_point
    eye, z_axis = start_state.pose[:3, 3], start_state.pose[:3, 2]
    radius = jp.linalg.norm(eye - start_state.target)

    ratio = jp.exp(dy / (0.5 * start_state.size[1]))
    translation = (1.0 - ratio) * radius * z_axis
    zoom_transform = SE3.translation(translation)

    new_pose = zoom_transform @ start_state.pose
    return start_state._replace(pose=new_pose)


def start_drag(state, point):
    """Initializes a drag operation."""
    return DragData(
        start_point=jp.array(point, dtype=jp.float32), start_state=state
    )


def drag(drag_data, current_point, mode):
    """Computes a new TrackballState based on a drag motion."""
    start_state = drag_data.start_state
    start_point = drag_data.start_point
    current_point = jp.array(current_point, dtype=jp.float32)

    cases = [
        lambda: _apply_rotation(start_state, start_point, current_point),
        lambda: _apply_pan(start_state, start_point, current_point),
        lambda: _apply_roll(start_state, start_point, current_point),
        lambda: _apply_zoom(start_state, start_point, current_point),
    ]
    return jax.lax.switch(mode, cases)


def scroll(state, clicks):
    """Computes a new state by zooming with the scroll wheel."""
    ratio = 0.95

    # FIX: Use explicit if/elif to correctly handle positive and negative
    # clicks, ensuring the zoom direction is intuitive.
    multiplier = 1.0
    if clicks > 0:
        multiplier = ratio**clicks
    elif clicks < 0:
        multiplier = (1.0 / ratio) ** abs(clicks)

    eye, z_axis = state.pose[:3, 3], state.pose[:3, 2]
    radius = jp.linalg.norm(eye - state.target)

    translation = (multiplier * radius - radius) * z_axis
    zoom_transform = SE3.translation(translation)

    new_pose = zoom_transform @ state.pose
    return state._replace(pose=new_pose)
