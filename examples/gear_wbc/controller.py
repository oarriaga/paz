"""PlayStation controller input for the PAZ GEAR-WBC demonstration.

The axis order is the one SDL reports for the Sony Wireless Controller on
Linux: both sticks, both analog triggers, and the d-pad as a hat.
"""

import os

# SDL pumps joystick state from the video event loop, so the dummy driver
# keeps the pad readable headless. It never opens a window of its own.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from simulation import Command
from simulation import DEFAULT_HEIGHT

LEFT_X, LEFT_Y, LEFT_TRIGGER = 0, 1, 2
RIGHT_X, RIGHT_Y, RIGHT_TRIGGER = 3, 4, 5
NUM_AXES = 6

DEADZONE = 0.1
MAX_FORWARD_SPEED = 0.5
MAX_LATERAL_SPEED = 0.4
MAX_YAW_RATE = 1.0
HEIGHT_RANGE = 0.15
MAX_TORSO_ANGLE = np.deg2rad(20)

CONTROLS = """GEAR-WBC / PAZ controller
  left stick         forward and lateral velocity
  right stick x      yaw rate
  right stick y      base height
  L2 / R2            torso roll
  d-pad up / down    torso pitch
  d-pad left / right torso yaw

Every input self-centers: release the pad to stand at the default pose.
"""


def build_pad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("Connect a PlayStation controller and rerun.")
    return pygame.joystick.Joystick(0)


def read_command(pad):
    pygame.event.pump()
    axes = [pad.get_axis(index) for index in range(NUM_AXES)]
    return compute_command(axes, pad.get_hat(0))


def compute_command(axes, hat):
    height_offset = apply_deadzone(axes[RIGHT_Y]) * HEIGHT_RANGE
    velocity = compute_velocity(axes)
    orientation = compute_orientation(axes, hat)
    return Command(velocity, DEFAULT_HEIGHT - height_offset, orientation)


def compute_velocity(axes):
    forward = apply_deadzone(-axes[LEFT_Y]) * MAX_FORWARD_SPEED
    lateral = apply_deadzone(-axes[LEFT_X]) * MAX_LATERAL_SPEED
    yaw_rate = apply_deadzone(-axes[RIGHT_X]) * MAX_YAW_RATE
    return np.array([forward, lateral, yaw_rate], "float32")


def compute_orientation(axes, hat):
    right_trigger = to_unit_range(axes[RIGHT_TRIGGER])
    left_trigger = to_unit_range(axes[LEFT_TRIGGER])
    hat_x, hat_y = hat
    angles = right_trigger - left_trigger, hat_y, -hat_x
    return (np.array(angles) * MAX_TORSO_ANGLE).astype("float32")


def apply_deadzone(value):
    if abs(value) < DEADZONE:
        scaled = 0.0
    else:
        scaled = (value - np.sign(value) * DEADZONE) / (1.0 - DEADZONE)
    return scaled


def to_unit_range(value):
    return (value + 1.0) / 2.0
