import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from collections import namedtuple
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from policy import FRAME_DIM
from policy import NUM_HISTORY_FRAMES
from policy import NUM_JOINTS
from policy import OBSERVATION_DIM
from simulation import ACTION_SCALE
from simulation import ANGULAR_VELOCITY_SCALE
from simulation import CONTROL_DECIMATION
from simulation import DAMPINGS
from simulation import DEFAULT_ANGLES
from simulation import GAINS
from simulation import JOINT_VELOCITY_SCALE
from simulation import MAX_BACKWARD_SPEED
from simulation import MAX_FORWARD_SPEED
from simulation import MAX_LATERAL_SPEED
from simulation import MAX_YAW_RATE
from simulation import SDK_INDICES
from simulation import SIMULATION_STEP
from simulation import TERM_OFFSETS
from simulation import build_history
from simulation import build_observation
from simulation import build_observation_frame
from simulation import compute_control
from simulation import compute_gravity_direction
from simulation import compute_joint_positions
from simulation import compute_target_angles
from simulation import FALL_HEIGHT
from simulation import compute_tilt
from simulation import draw_push
from simulation import has_fallen
from simulation import sample_heading
from simulation import update_history

TERM_NAMES = ["base_ang_vel", "projected_gravity", "velocity_commands",
              "joint_pos_rel", "joint_vel_rel", "last_action"]

EXPERIMENT = Path(__file__).parent / "unitree_g1_29dof_velocity_robust"

FakeData = namedtuple("FakeData", "qpos qvel")


def build_fake_data(seed=0):
    rng = np.random.default_rng(seed)
    qpos = rng.normal(size=7 + NUM_JOINTS)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qvel = rng.normal(size=6 + NUM_JOINTS)
    return FakeData(qpos, qvel)


def load_deploy_configuration():
    paths = sorted(EXPERIMENT.glob("*/params/deploy.yaml"))
    return yaml.safe_load(paths[-1].read_text())


needs_checkpoint = pytest.mark.skipif(
    not list(EXPERIMENT.glob("*/params/deploy.yaml")),
    reason="no unitree_rl_lab run directory next to this example")


def test_observation_frame_places_every_term_at_its_trained_offset():
    data = build_fake_data()
    command = np.array([0.5, 0.1, -0.2], "float32")
    action = np.arange(NUM_JOINTS, dtype="float32")
    frame = build_observation_frame(data, command, action)
    assert frame.shape == (FRAME_DIM,)
    angular = data.qvel[3:6] * ANGULAR_VELOCITY_SCALE
    assert np.allclose(frame[0:3], angular)
    assert np.allclose(frame[3:6], [0.0, 0.0, -1.0], atol=1e-6)
    assert np.allclose(frame[6:9], command)
    joints = (data.qpos[7:7 + NUM_JOINTS] - DEFAULT_ANGLES)[SDK_INDICES]
    assert np.allclose(frame[9:38], joints, atol=1e-6)
    velocities = data.qvel[6:6 + NUM_JOINTS] * JOINT_VELOCITY_SCALE
    assert np.allclose(frame[38:67], velocities[SDK_INDICES], atol=1e-6)
    assert np.allclose(frame[67:96], action)


def test_joint_permutation_pairs_the_two_legs_in_the_trained_order():
    data = build_fake_data()
    positions = compute_joint_positions(data)
    raw = data.qpos[7:7 + NUM_JOINTS] - DEFAULT_ANGLES
    assert positions[0] == pytest.approx(raw[0])
    assert positions[1] == pytest.approx(raw[6])
    assert positions[2] == pytest.approx(raw[12])


def test_target_angles_scatter_back_to_the_mujoco_joint_order():
    action = np.arange(NUM_JOINTS, dtype="float32")
    targets = compute_target_angles(action)
    assert np.allclose(targets[SDK_INDICES], action * ACTION_SCALE
                       + DEFAULT_ANGLES[SDK_INDICES])


def test_target_angles_of_a_zero_action_are_the_default_pose():
    targets = compute_target_angles(np.zeros(NUM_JOINTS, "float32"))
    assert np.allclose(targets, DEFAULT_ANGLES)


def test_gravity_direction_is_down_for_an_upright_base():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(compute_gravity_direction(upright), [0, 0, -1], 1e-6)


def test_gravity_direction_tilts_with_a_pitched_base():
    half = np.sqrt(0.5)
    pitched = np.array([half, 0.0, half, 0.0])
    direction = compute_gravity_direction(pitched)
    assert np.allclose(direction, [1.0, 0.0, 0.0], atol=1e-6)


def test_tilt_grows_from_zero_as_the_base_rolls_over():
    upright = build_fake_data()
    assert compute_tilt(upright) == pytest.approx(0.0, abs=1e-6)
    half = np.sqrt(0.5)
    upright.qpos[3:7] = [half, 0.0, half, 0.0]
    assert compute_tilt(upright) == pytest.approx(np.pi / 2, abs=1e-6)


def test_history_repeats_the_first_frame_the_way_a_reset_does():
    frame = np.arange(FRAME_DIM, dtype="float32")
    history = build_history(frame)
    assert history.shape == (NUM_HISTORY_FRAMES, FRAME_DIM)
    assert np.allclose(history, frame)


def test_history_drops_the_oldest_frame_and_keeps_the_newest_last():
    history = build_history(np.zeros(FRAME_DIM, "float32"))
    for index in range(1, NUM_HISTORY_FRAMES + 2):
        history = update_history(history, np.full(FRAME_DIM, index, "f4"))
    assert np.allclose(history[0], 2.0)
    assert np.allclose(history[-1], NUM_HISTORY_FRAMES + 1)


def test_observation_groups_every_term_history_instead_of_whole_frames():
    frames = np.arange(NUM_HISTORY_FRAMES * FRAME_DIM, dtype="float32")
    history = frames.reshape(NUM_HISTORY_FRAMES, FRAME_DIM)
    observation = build_observation(history)
    assert observation.shape == (1, OBSERVATION_DIM)
    assert np.allclose(observation[0, 0:3], history[0, 0:3])
    assert np.allclose(observation[0, 3:6], history[1, 0:3])
    assert np.allclose(observation[0, 12:15], history[4, 0:3])
    assert np.allclose(observation[0, 15:18], history[0, 3:6])


def test_observation_terms_have_the_trained_widths():
    widths = np.diff((0,) + TERM_OFFSETS + (FRAME_DIM,))
    assert widths.tolist() == [3, 3, 3, 29, 29, 29]


def test_control_opposes_position_error_and_joint_velocity():
    data = build_fake_data()
    targets = data.qpos[7:7 + NUM_JOINTS].astype("float32")
    control = compute_control(data, targets)
    expected = -data.qvel[6:6 + NUM_JOINTS] * DAMPINGS
    assert np.allclose(control, expected, atol=1e-4)


def build_pushed_body():
    xml = ("<mujoco><worldbody><body name='torso'><freejoint/>"
           "<geom size='0.1'/></body></worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    return mujoco.MjData(model), mujoco.MjvScene(model, 10)


def test_push_arrow_is_drawn_only_while_a_shove_is_applied():
    data, scene = build_pushed_body()
    draw_push(scene, data, 1)
    assert scene.ngeom == 0
    data.xfrc_applied[1, 0:3] = [300.0, 0.0, 0.0]
    draw_push(scene, data, 1)
    assert scene.ngeom == 1


def test_push_arrow_length_grows_with_the_force():
    data, scene = build_pushed_body()
    data.xfrc_applied[1, 0:3] = [100.0, 0.0, 0.0]
    draw_push(scene, data, 1)
    short = scene.geoms[0].size[2]
    data.xfrc_applied[1, 0:3] = [300.0, 0.0, 0.0]
    draw_push(scene, data, 1)
    assert scene.geoms[0].size[2] == pytest.approx(3 * short)


@needs_checkpoint
def test_constants_match_the_exported_deployment_configuration():
    config = load_deploy_configuration()
    assert config["joint_ids_map"] == SDK_INDICES.tolist()
    assert config["stiffness"] == GAINS.tolist()
    assert config["damping"] == DAMPINGS.tolist()
    assert config["step_dt"] == SIMULATION_STEP * CONTROL_DECIMATION
    defaults = np.array(config["default_joint_pos"])
    assert np.allclose(DEFAULT_ANGLES[SDK_INDICES], defaults)


@needs_checkpoint
def test_action_scale_and_offset_match_the_exported_configuration():
    action = load_deploy_configuration()["actions"]["JointPositionAction"]
    defaults = np.array(load_deploy_configuration()["default_joint_pos"])
    assert action["scale"] == [ACTION_SCALE] * NUM_JOINTS
    assert np.allclose(action["offset"], defaults)


@needs_checkpoint
def test_observation_terms_and_scales_match_the_exported_configuration():
    terms = load_deploy_configuration()["observations"]
    assert list(terms) == TERM_NAMES
    lengths = [len(np.atleast_1d(term["scale"])) for term in terms.values()]
    assert lengths == [3, 3, 3, 29, 29, 29]
    assert terms["base_ang_vel"]["scale"][0] == ANGULAR_VELOCITY_SCALE
    assert terms["joint_vel_rel"]["scale"][0] == JOINT_VELOCITY_SCALE
    histories = [term["history_length"] for term in terms.values()]
    assert histories == [NUM_HISTORY_FRAMES] * len(TERM_NAMES)


@needs_checkpoint
def test_speed_limits_match_the_exported_command_ranges():
    ranges = load_deploy_configuration()["commands"]["base_velocity"]
    ranges = ranges["ranges"]
    assert ranges["lin_vel_x"] == [-MAX_BACKWARD_SPEED, MAX_FORWARD_SPEED]
    assert ranges["lin_vel_y"] == [-MAX_LATERAL_SPEED, MAX_LATERAL_SPEED]
    assert ranges["ang_vel_z"] == [-MAX_YAW_RATE, MAX_YAW_RATE]


def test_fall_is_detected_by_a_dropped_base_as_well_as_by_tilt():
    data = build_fake_data()
    data.qpos[2] = 0.8
    assert not has_fallen(data)
    data.qpos[2] = FALL_HEIGHT - 0.01
    assert has_fallen(data)


def test_sampled_heading_is_a_unit_yaw_quaternion_in_mujoco_order():
    rng = np.random.default_rng(0)
    for _ in range(8):
        heading = sample_heading(rng)
        assert np.linalg.norm(heading) == pytest.approx(1.0)
        assert np.allclose(heading[1:3], 0.0)
        direction = compute_gravity_direction(heading)
        assert np.allclose(direction, [0.0, 0.0, -1.0], atol=1e-6)


def test_sampled_headings_differ_between_seeds():
    first = sample_heading(np.random.default_rng(0))
    assert not np.allclose(first, sample_heading(np.random.default_rng(1)))
