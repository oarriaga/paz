import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from controller import DEADZONE
from controller import HEIGHT_RANGE
from controller import LEFT_X
from controller import LEFT_Y
from controller import MAX_FORWARD_SPEED
from controller import MAX_LATERAL_SPEED
from controller import MAX_TORSO_ANGLE
from controller import MAX_YAW_RATE
from controller import RIGHT_TRIGGER
from controller import RIGHT_X
from controller import RIGHT_Y
from controller import apply_deadzone
from controller import compute_command
from simulation import DEFAULT_HEIGHT

CENTERED = 0, 0


def build_released_axes():
    return [0.0, 0.0, -1.0, 0.0, 0.0, -1.0]


def test_released_pad_commands_the_default_standing_pose():
    command = compute_command(build_released_axes(), CENTERED)
    assert np.allclose(command.velocity, 0.0)
    assert command.height == pytest.approx(DEFAULT_HEIGHT)
    assert np.allclose(command.orientation, 0.0)


def test_resting_stick_offset_stays_inside_the_deadzone():
    axes = build_released_axes()
    axes[LEFT_Y] = -0.05
    axes[RIGHT_Y] = 0.05
    command = compute_command(axes, CENTERED)
    assert np.allclose(command.velocity, 0.0)
    assert command.height == pytest.approx(DEFAULT_HEIGHT)


def test_left_stick_forward_commands_forward_velocity_alone():
    axes = build_released_axes()
    axes[LEFT_Y] = -1.0
    command = compute_command(axes, CENTERED)
    assert command.velocity[0] == pytest.approx(MAX_FORWARD_SPEED)
    assert np.allclose(command.velocity[1:], 0.0)


def test_left_stick_left_commands_positive_lateral_velocity():
    axes = build_released_axes()
    axes[LEFT_X] = -1.0
    command = compute_command(axes, CENTERED)
    assert command.velocity[1] == pytest.approx(MAX_LATERAL_SPEED)


def test_right_stick_left_commands_a_positive_yaw_rate():
    axes = build_released_axes()
    axes[RIGHT_X] = -1.0
    command = compute_command(axes, CENTERED)
    assert command.velocity[2] == pytest.approx(MAX_YAW_RATE)


def test_right_stick_down_lowers_the_commanded_base_height():
    axes = build_released_axes()
    axes[RIGHT_Y] = 1.0
    command = compute_command(axes, CENTERED)
    assert command.height == pytest.approx(DEFAULT_HEIGHT - HEIGHT_RANGE)


def test_right_trigger_rolls_the_torso_to_its_right():
    axes = build_released_axes()
    axes[RIGHT_TRIGGER] = 1.0
    command = compute_command(axes, CENTERED)
    assert command.orientation[0] == pytest.approx(MAX_TORSO_ANGLE)
    assert np.allclose(command.orientation[1:], 0.0)


def test_hat_commands_torso_pitch_forward_and_yaw_toward_its_right():
    axes = build_released_axes()
    pitched = compute_command(axes, (0, 1))
    yawed = compute_command(axes, (1, 0))
    assert pitched.orientation[1] == pytest.approx(MAX_TORSO_ANGLE)
    assert yawed.orientation[2] == pytest.approx(-MAX_TORSO_ANGLE)


def test_deadzone_rescales_so_full_deflection_still_reaches_one():
    assert apply_deadzone(DEADZONE / 2) == pytest.approx(0.0)
    assert apply_deadzone(1.0) == pytest.approx(1.0)
    assert apply_deadzone(-1.0) == pytest.approx(-1.0)
