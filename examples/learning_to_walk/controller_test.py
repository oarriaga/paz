import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from controller import DEADZONE
from controller import LEFT_X
from controller import LEFT_Y
from controller import NUM_AXES
from controller import RIGHT_X
from controller import apply_deadzone
from controller import compute_command
from simulation import MAX_BACKWARD_SPEED
from simulation import MAX_FORWARD_SPEED
from simulation import MAX_LATERAL_SPEED
from simulation import MAX_YAW_RATE


def build_released_axes():
    return [0.0] * NUM_AXES


def test_released_pad_commands_a_standstill():
    assert np.allclose(compute_command(build_released_axes()), 0.0)


def test_resting_stick_offset_stays_inside_the_deadzone():
    axes = build_released_axes()
    axes[LEFT_Y] = -0.05
    axes[RIGHT_X] = 0.05
    assert np.allclose(compute_command(axes), 0.0)


def test_left_stick_forward_commands_the_fastest_trained_walk():
    axes = build_released_axes()
    axes[LEFT_Y] = -1.0
    command = compute_command(axes)
    assert command[0] == pytest.approx(MAX_FORWARD_SPEED)
    assert np.allclose(command[1:], 0.0)


def test_left_stick_back_is_capped_lower_than_forward():
    axes = build_released_axes()
    axes[LEFT_Y] = 1.0
    assert compute_command(axes)[0] == pytest.approx(-MAX_BACKWARD_SPEED)


def test_left_stick_left_commands_positive_lateral_velocity():
    axes = build_released_axes()
    axes[LEFT_X] = -1.0
    assert compute_command(axes)[1] == pytest.approx(MAX_LATERAL_SPEED)


def test_right_stick_left_commands_a_positive_yaw_rate():
    axes = build_released_axes()
    axes[RIGHT_X] = -1.0
    assert compute_command(axes)[2] == pytest.approx(MAX_YAW_RATE)


def test_deadzone_rescales_so_full_deflection_still_reaches_one():
    assert apply_deadzone(DEADZONE / 2) == pytest.approx(0.0)
    assert apply_deadzone(1.0) == pytest.approx(1.0)
    assert apply_deadzone(-1.0) == pytest.approx(-1.0)
