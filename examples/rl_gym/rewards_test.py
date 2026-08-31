import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import rewards


def test_linear_velocity_tracking():
    command = jp.array([0.5, 0.0, 0.0])
    velocity = jp.array([0.3, 0.1, 0.4])
    computed = float(rewards.linear_velocity_tracking(velocity, command))
    reference = np.exp(-(0.2**2 + 0.1**2) / 0.25)
    assert np.isclose(computed, reference, atol=1e-5)


def test_angular_velocity_tracking():
    command = jp.array([0.0, 0.0, 0.2])
    angular = jp.array([1.0, 1.0, -0.1])
    computed = float(rewards.angular_velocity_tracking(angular, command))
    assert np.isclose(computed, np.exp(-(0.3**2) / 0.25), atol=1e-5)


def test_tracking_is_one_at_zero_error():
    command = jp.array([0.4, -0.2, 0.1])
    velocity = jp.array([0.4, -0.2, 0.0])
    assert np.isclose(float(rewards.linear_velocity_tracking(velocity, command)), 1.0)  # fmt: skip


def test_gait_match_agreement_and_standing_gate():
    # at step 0 both phases are inside stance, so both feet should touch
    contact = jp.array([True, True])
    command = jp.array([0.5, 0.0, 0.0])
    assert float(rewards.gait_match(0, contact, command)) == 2.0
    swinging = jp.array([False, False])
    assert float(rewards.gait_match(0, swinging, command)) == 0.0
    standing = jp.array([0.0, 0.0, 0.0])
    assert float(rewards.gait_match(0, contact, standing)) == 0.0


def test_gait_match_offset_phases():
    # at 0.6 of the period the first foot swings and the second stands
    step = int(0.6 * 0.8 / 0.02)
    contact = jp.array([False, True])
    command = jp.array([0.5, 0.0, 0.0])
    assert float(rewards.gait_match(step, contact, command)) == 2.0


def test_foot_clearance_is_one_at_target_height():
    heights = jp.array([0.1, 0.1])
    velocities = jp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    computed = float(rewards.foot_clearance(heights, velocities))
    assert np.isclose(computed, 1.0, atol=1e-5)


def test_joint_limit_violation():
    positions = jp.array([-1.5, 0.0, 2.0])
    lower, upper = jp.full(3, -1.0), jp.full(3, 1.0)
    computed = float(rewards.joint_limit_violation(positions, lower, upper))
    assert np.isclose(computed, 0.5 + 1.0, atol=1e-5)


def test_energy():
    velocities, torque = jp.array([1.0, -2.0]), jp.array([-3.0, 4.0])
    assert np.isclose(float(rewards.energy(velocities, torque)), 11.0)


def test_foot_slip_counts_contact_feet_only():
    velocities = jp.array([[3.0, 4.0, 9.0], [1.0, 0.0, 9.0]])
    contact = jp.array([1.0, 0.0])
    assert np.isclose(float(rewards.foot_slip(velocities, contact)), 5.0)


def test_compute_reward_weights_and_order():
    tracking = 1.0, 2.0
    base = 3.0, 4.0, 5.0, 6.0
    joint = 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
    feet = 15.0, 16.0, 17.0, 18.0
    stability = 19.0, 20.0
    args = tracking, joint, base, feet, 1.0, stability
    reward, values = rewards.compute_reward(*args)
    expected_order = [1.0, 2.0, 1.0, 3.0, 4.0, *joint, 5.0, 6.0, *feet, 19.0, 20.0]  # fmt: skip
    assert np.allclose(np.asarray(values), expected_order)
    expected = np.sum(np.asarray(values) * np.asarray(rewards.ROBUST_WEIGHTS))
    assert np.isclose(float(reward), expected * 0.02, atol=1e-4)


def test_alive_reward_stops_on_the_falling_step():
    tracking, base = (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    joint, feet = (0.0,) * 8, (0.0, 0.0, 0.0, 0.0)
    stability = 0.0, 0.0
    args = tracking, joint, base, feet
    standing, _ = rewards.compute_reward(*args, 1.0, stability)
    fallen, _ = rewards.compute_reward(*args, 0.0, stability)
    assert np.isclose(float(standing) - float(fallen), 0.15 * 0.02)


def test_weights_match_reference_configuration():
    # unitree_rl_lab weights in values order; the two postural stabilizer
    # slots stay in the array but carry no weight
    reference = [1.0, 0.5, 0.15, -2.0, -0.05, -0.001, -2.5e-7, -0.05, -5.0, -2e-5, -0.1, -1.0, -1.0, -5.0, -10.0, 0.5, -0.2, 1.0, -1.0, 0.0, 0.0]  # fmt: skip
    assert np.allclose(np.asarray(rewards.ROBUST_WEIGHTS), reference)


def test_upright_is_one_when_level():
    assert np.isclose(float(rewards.upright(jp.array([0.0, 0.0, -1.0]))), 1.0)  # fmt: skip
    tilted = jp.array([0.3, 0.4, -0.866])
    expected = np.exp(-(0.09 + 0.16) / 0.2)
    assert np.isclose(float(rewards.upright(tilted)), expected, atol=1e-5)


def test_posture_tolerance_switches_with_command_speed():
    defaults = jp.zeros(29)
    positions = defaults.at[3].set(0.2)
    standing = jp.array([0.0, 0.0, 0.0])
    walking = jp.array([0.5, 0.0, 0.0])
    strict = float(rewards.posture(positions, defaults, standing))
    loose = float(rewards.posture(positions, defaults, walking))
    assert np.isclose(strict, np.exp(-(0.2 / 0.05) ** 2 / 29), atol=1e-4)
    assert np.isclose(loose, np.exp(-(0.2 / 0.35) ** 2 / 29), atol=1e-4)
    assert loose > strict
