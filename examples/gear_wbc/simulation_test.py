import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from collections import namedtuple

import numpy as np
import pytest

from simulation import ACTION_SCALE
from simulation import BALANCE_SPEED
from simulation import DEFAULT_ANGLES
from simulation import JOINT_VELOCITY_SCALE
from simulation import LOWER_BODY_ANGLES
from simulation import NUM_JOINTS
from simulation import VELOCITY_COMMAND_SCALE
from simulation import build_command
from simulation import build_command_frame
from simulation import build_history
from simulation import build_observation_frame
from simulation import compute_gravity_direction
from simulation import compute_target_angles
from simulation import compute_torques
from simulation import select_actor
from simulation import update_history

FakeData = namedtuple("FakeData", "qpos qvel")


def build_fake_data(seed=0):
    rng = np.random.default_rng(seed)
    qpos = rng.normal(size=7 + NUM_JOINTS)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qvel = rng.normal(size=6 + NUM_JOINTS)
    return FakeData(qpos, qvel)


def test_observation_frame_places_every_field_at_its_release_offset():
    data = build_fake_data()
    command = build_command(0.0)
    action = np.arange(15, dtype="float32")
    frame = build_observation_frame(data, command, action)
    assert frame.shape == (86,)
    assert np.allclose(frame[0:3], command.velocity * VELOCITY_COMMAND_SCALE)
    assert frame[3] == pytest.approx(command.height)
    assert np.allclose(frame[4:7], command.orientation)
    assert np.allclose(frame[10:13], [0.0, 0.0, -1.0], atol=1e-6)
    joints = data.qpos[7:7 + NUM_JOINTS] - DEFAULT_ANGLES
    assert np.allclose(frame[13:42], joints, atol=1e-6)
    velocities = data.qvel[6:6 + NUM_JOINTS] * JOINT_VELOCITY_SCALE
    assert np.allclose(frame[42:71], velocities, atol=1e-6)
    assert np.allclose(frame[71:86], action)


def test_command_frame_scales_velocity_but_not_height_or_orientation():
    velocity = np.array([1.0, 1.0, 1.0], "float32")
    orientation = np.array([0.1, 0.2, 0.3], "float32")
    fields = dict(velocity=velocity, height=0.5, orientation=orientation)
    command = build_command(0.0)._replace(**fields)
    frame = build_command_frame(command)
    assert np.allclose(frame[0:3], VELOCITY_COMMAND_SCALE)
    assert frame[3] == pytest.approx(0.5)
    assert np.allclose(frame[4:7], orientation)


def test_gravity_direction_is_down_for_an_upright_base():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    direction = compute_gravity_direction(upright)
    assert np.allclose(direction, [0.0, 0.0, -1.0], atol=1e-6)


def test_gravity_direction_tilts_with_a_pitched_base():
    half = np.sqrt(0.5)
    pitched = np.array([half, 0.0, half, 0.0])
    direction = compute_gravity_direction(pitched)
    assert np.allclose(direction, [1.0, 0.0, 0.0], atol=1e-6)


def test_history_starts_zeroed_and_keeps_the_newest_frame_last():
    history = build_history()
    frame = np.arange(86, dtype="float32")
    observation = update_history(history, frame)
    assert observation.shape == (1, 516)
    assert np.allclose(observation[0, :-86], 0.0)
    assert np.allclose(observation[0, -86:], frame)


def test_history_drops_the_oldest_frame_after_six_updates():
    history = build_history()
    for index in range(7):
        frame = np.full(86, index, dtype="float32")
        observation = update_history(history, frame)
    assert np.allclose(observation[0, :86], 1.0)
    assert np.allclose(observation[0, -86:], 6.0)


def test_select_actor_switches_experts_at_the_release_threshold():
    models = namedtuple("Models", "balance walk")("balance", "walk")
    standing = build_command(0.0)
    assert select_actor(models, standing) == "balance"
    fast = np.array([BALANCE_SPEED + 0.01, 0.0, 0.0], "float32")
    assert select_actor(models, standing._replace(velocity=fast)) == "walk"


def test_target_angles_offset_the_scaled_action_from_the_default_pose():
    action = np.ones(15, dtype="float32")
    targets = compute_target_angles(action)
    assert np.allclose(targets, LOWER_BODY_ANGLES + ACTION_SCALE)


def test_torques_oppose_position_error_and_velocity():
    args = np.zeros(2), np.array([1.0, -1.0]), np.array([1.0, 1.0])
    torques = compute_torques(*args, 10.0, 2.0)
    assert np.allclose(torques, [-12.0, 8.0])
