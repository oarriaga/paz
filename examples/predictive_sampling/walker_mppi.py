import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import jax.numpy as jp
import mujoco

from controller import PredictiveSampler
from samplers import MPPISampler
from spline import interpolate_zero
from structures import Task
from task import Model
from viewer import run_interactive
from walker import WalkerArgs
from walker import build_walker_sensors
from walker import get_torso_velocity


MODEL_PATH = Path(__file__).parent / "models" / "walker" / "scene.xml"

COMMANDED_VELOCITY = 1.2
TARGET_HEIGHT = 0.0


def WalkerMPPI(impl="jax"):
    model_kwargs = dict(trace_sites=("torso_site",), backend=impl)
    model_kwargs["naconmax"] = 800
    model = Model(MODEL_PATH, **model_kwargs)
    sensors = build_walker_sensors(model.mujoco_model)
    targets = COMMANDED_VELOCITY, TARGET_HEIGHT
    walker_args = WalkerArgs(model.model, *sensors, *targets)
    running = partial(running_cost, walker_args)
    terminal = partial(terminal_cost, walker_args)
    return Task(running, terminal), model


def running_cost(args, state, control):
    return terminal_cost(args, state)


def terminal_cost(args, state):
    velocity = get_torso_velocity(args, state) - args.target_velocity
    return jp.square(velocity)


def build_simulation_model(model):
    mujoco_model = deepcopy(model.mujoco_model)
    mujoco_model.opt.timestep = 0.005
    mujoco_model.opt.iterations = 50
    return mujoco_model


parser = argparse.ArgumentParser()
parser.add_argument("--warp", action="store_true")
args = parser.parse_args()

task, model = WalkerMPPI(impl="warp" if args.warp else "jax")
sampler = MPPISampler(model, 256, 5, 1.0, 0.5, 0.05, interpolate_zero)
controller = PredictiveSampler(task, model, sampler, 1)

mujoco_model = build_simulation_model(model)
data = mujoco.MjData(mujoco_model)
parameters = controller.sampler.initialize()
viewer_args = controller, mujoco_model, data, parameters
viewer_kwargs = dict(frequency=50, fixed_camera_id=0, show_traces=False)
viewer_kwargs["max_traces"] = 1
run_interactive(*viewer_args, **viewer_kwargs)
