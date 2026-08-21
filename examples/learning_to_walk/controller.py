"""PlayStation controller input for the learning-to-walk demonstration.

The axis order is the one SDL reports for the Sony Wireless Controller on
Linux. The speed limits are the command limit_ranges the policy was trained
against, so a fully deflected stick asks for exactly what it has seen.
"""

import os

# SDL pumps joystick state from the video event loop, so the dummy driver
# keeps the pad readable headless. It never opens a window of its own.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from simulation import MAX_BACKWARD_SPEED
from simulation import MAX_FORWARD_SPEED
from simulation import MAX_LATERAL_SPEED
from simulation import MAX_YAW_RATE

LEFT_X, LEFT_Y = 0, 1
RIGHT_X = 3
NUM_AXES = 4
PUSH_BUTTON = 0

DEADZONE = 0.1

CONTROLS = """unitree_rl_lab G1 velocity policy / PAZ
  left stick up      forward velocity, up to 1.0 m/s
  left stick down    backward velocity, up to 0.5 m/s
  left stick sides   lateral velocity, up to 0.3 m/s
  right stick sides  yaw rate, up to 0.2 rad/s
  cross              shove the torso

Every stick self-centers: release the pad to command a standstill.
"""


def find_pad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        pad = None
    else:
        pad = pygame.joystick.Joystick(0)
        if pad.get_numaxes() < NUM_AXES:
            pad = None
    return pad


def read_command(pad):
    pygame.event.pump()
    return compute_command([pad.get_axis(i) for i in range(NUM_AXES)])


def read_push(pad):
    return pad.get_button(PUSH_BUTTON) == 1


def compute_command(axes):
    forward = compute_forward_speed(apply_deadzone(-axes[LEFT_Y]))
    lateral = apply_deadzone(-axes[LEFT_X]) * MAX_LATERAL_SPEED
    yaw_rate = apply_deadzone(-axes[RIGHT_X]) * MAX_YAW_RATE
    return np.array([forward, lateral, yaw_rate], "float32")


def compute_forward_speed(value):
    if value >= 0.0:
        speed = value * MAX_FORWARD_SPEED
    else:
        speed = value * MAX_BACKWARD_SPEED
    return speed


def apply_deadzone(value):
    if abs(value) < DEADZONE:
        scaled = 0.0
    else:
        scaled = (value - np.sign(value) * DEADZONE) / (1.0 - DEADZONE)
    return scaled
