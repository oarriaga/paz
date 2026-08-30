import jax.numpy as jp
import paz

ROBUST_WEIGHTS = jp.array([1.0, 0.5, 0.15, -2.0, -0.05, -0.001, -2.5e-7, -0.05, -5.0, -2e-5, -0.1, -1.0, -1.0, -5.0, -10.0, 0.5, -0.2, 1.0, -1.0])  # fmt: skip


# Rewards: bigger is better, non-negative, entered with a positive weight.


def linear_velocity_tracking(linear_velocity, command, stdv=0.5):
    args = command[:2], linear_velocity[:2]
    squared_error = paz.losses.mse(*args, reduction="sum")
    return exponential_decay(squared_error, stdv)


def angular_velocity_tracking(angular_velocity, command, stdv=0.5):
    args = command[2], angular_velocity[2]
    squared_error = paz.losses.mse(*args, reduction="sum")
    return exponential_decay(squared_error, stdv)


def touchdown_accuracy(touchdown, error, stdv=0.10):
    return jp.sum(touchdown * exponential_decay(error**2, stdv))


def gait_match(step, contact, command, control_step=0.02, gait_period=0.8, stance_threshold=0.55, minimum_speed=0.1):  # fmt: skip
    phase = (step * control_step % gait_period) / gait_period
    phases = jp.mod(phase + jp.array([0.0, 0.5]), 1.0)
    stance = phases < stance_threshold
    commanded = jp.linalg.norm(command) > minimum_speed
    return jp.sum(stance == contact) * commanded


def foot_clearance(heights, velocities, clearance_height=0.1, clearance_stdv=0.05, speed_scale=2.0):  # fmt: skip
    height_error = (heights - clearance_height) ** 2
    speed = jp.tanh(speed_scale * jp.linalg.norm(velocities[:, :2], axis=-1))
    return jp.exp(-jp.sum(height_error * speed) / clearance_stdv)


def exponential_decay(squared_error, stdv):
    # a Gaussian likelihood over its own peak, so zero error scores exactly 1
    # not a density: no normalizer, and the width is stdv / sqrt(2), not stdv
    return jp.exp(-squared_error / stdv**2)


# Costs: smaller is better, non-negative, entered with a negative weight.


def vertical_velocity_error(linear_velocity):
    return linear_velocity[2] ** 2


def roll_pitch_rate_error(angular_velocity):
    return jp.sum(angular_velocity[:2] ** 2)


def tilt_error(gravity):
    return jp.sum(gravity[:2] ** 2)


def base_height_error(height, height_target=0.76):
    return paz.losses.mse(height, height_target, reduction="sum")


def joint_velocity_error(velocities):
    return jp.sum(velocities**2)


def joint_acceleration_error(accelerations):
    return jp.sum(accelerations**2)


def action_rate_error(action, last_action):
    return paz.losses.mse(action, last_action, reduction="sum")


def joint_limit_violation(positions, min_position, max_position):
    lower_error = jp.clip(positions - min_position, max=0.0)
    upper_error = jp.clip(positions - max_position, min=0.0)
    return jp.sum(-lower_error + upper_error)


def arm_deviation(positions, defaults, arms):
    return paz.losses.mae(positions[arms], defaults[arms], reduction="sum")


def waist_deviation(positions, defaults, waists):
    return paz.losses.mae(positions[waists], defaults[waists], reduction="sum")


def hip_deviation(positions, defaults, hips):
    return paz.losses.mae(positions[hips], defaults[hips], reduction="sum")


def energy(velocities, torque):
    return jp.sum(jp.abs(velocities) * jp.abs(torque))


def foot_slip(velocities, contact):
    return jp.sum(jp.linalg.norm(velocities[:, :2], axis=-1) * contact)


# Term groups: each returns the values one weight block of ROBUST_WEIGHTS
# multiplies, in that block's order.


def compute_tracking_terms(yaw_velocity, angular_velocity, command):
    linear = linear_velocity_tracking(yaw_velocity, command)
    return linear, angular_velocity_tracking(angular_velocity, command)


def compute_base_terms(linear_velocity, angular_velocity, gravity, height):
    vertical = vertical_velocity_error(linear_velocity)
    roll_pitch = roll_pitch_rate_error(angular_velocity)
    return vertical, roll_pitch, tilt_error(gravity), base_height_error(height)


def compute_joint_terms(positions, velocities, accelerations, torque, action, last_action, defaults, limits, indices):  # fmt: skip
    lower, upper = limits
    velocity_error = joint_velocity_error(velocities)
    acceleration_error = joint_acceleration_error(accelerations)
    rate_error = action_rate_error(action, last_action)
    limit_error = joint_limit_violation(positions, lower, upper)
    power = energy(velocities, torque)
    arms = arm_deviation(positions, defaults, indices.arms)
    waists = waist_deviation(positions, defaults, indices.waists)
    hips = hip_deviation(positions, defaults, indices.hips)
    return velocity_error, acceleration_error, rate_error, limit_error, power, arms, waists, hips  # fmt: skip


def compute_foot_terms(step, contact, velocities, heights, command, undesired):  # fmt: skip
    gait = gait_match(step, contact, command)
    slip = foot_slip(velocities, contact)
    return gait, slip, foot_clearance(heights, velocities), undesired


def compute_reward(tracking, joint, base, feet, alive, control_step=0.02):
    linear, angular = tracking
    vertical, roll_pitch, tilt, height = base
    values = jp.array([linear, angular, alive, vertical, roll_pitch, *joint, tilt, height, *feet])  # fmt: skip
    return jp.sum(values * ROBUST_WEIGHTS) * control_step, values
