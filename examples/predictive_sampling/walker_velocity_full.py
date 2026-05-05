import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import jax.numpy as jp
import mujoco

from controller import PredictiveSampler
from samplers import KnotSampler
from spline import interpolate_zero
from structures import Task
from task import Model
from viewer import run_interactive
from walker import WalkerArgs
from walker import build_walker_sensors
from walker import get_torso_deviation
from walker import get_torso_height
from walker import get_torso_velocity


MODEL_PATH = Path(__file__).parent / "models" / "walker" / "scene.xml"

COMMANDED_VELOCITY = 1.2
TARGET_HEIGHT = 1.2

W_VEL = 1.0
W_HEIGHT = 10.0
W_UPRIGHT = 3.0
W_PITCH_RATE = 0.1
W_CONTROL = 0.1


def WalkerVelocityFull(impl="jax"):
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
    state_cost = terminal_cost(args, state)
    control_cost = W_CONTROL * jp.sum(jp.square(control))
    return state_cost + control_cost


def terminal_cost(args, state):
    velocity = get_torso_velocity(args, state) - args.target_velocity
    height = get_torso_height(args, state) - args.target_height
    upright = get_torso_deviation(args, state)
    pitch_rate = state.qvel[2]
    cost = W_VEL * jp.square(velocity)
    cost += W_HEIGHT * jp.square(height)
    cost += W_UPRIGHT * jp.square(upright)
    cost += W_PITCH_RATE * jp.square(pitch_rate)
    return cost


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", action="store_true")
    return parser.parse_args()


def build_controller(task, model):
    sampler_args = model, 256, 5, 1.0, 0.5, interpolate_zero
    sampler = KnotSampler(*sampler_args)
    return PredictiveSampler(task, model, sampler, 1)


def build_simulation_model(model):
    mujoco_model = deepcopy(model.mujoco_model)
    mujoco_model.opt.timestep = 0.005
    mujoco_model.opt.iterations = 50
    return mujoco_model


def main():
    args = parse_args()
    task, model = WalkerVelocityFull(impl="warp" if args.warp else "jax")
    controller = build_controller(task, model)
    mujoco_model = build_simulation_model(model)
    mj_data = mujoco.MjData(mujoco_model)
    parameters = controller.sampler.initialize()
    viewer_args = controller, mujoco_model, mj_data, parameters
    viewer_kwargs = dict(frequency=50, fixed_camera_id=0, show_traces=False)
    viewer_kwargs["max_traces"] = 1
    run_interactive(*viewer_args, **viewer_kwargs)


if __name__ == "__main__":
    main()
