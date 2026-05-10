import argparse
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path

import jax.numpy as jp
import mujoco

from controller import PredictiveSampler
from samplers import KnotSampler
from spline import interpolate_cubic
from task import Model
from viewer import run_interactive

MODEL_PATH = Path(__file__).parent / "models" / "walker" / "scene.xml"

walker_arg_fields = "model torso_position_sensor torso_velocity_sensor"
walker_arg_fields += " torso_zaxis_sensor target_velocity target_height"
WalkerArgs = namedtuple("WalkerArgs", walker_arg_fields)
Task = namedtuple("Task", "running_cost")


def Walker(impl="jax", target_velocity=0.0, target_height=1.2):
    model_kwargs = dict(trace_sites=("torso_site",), backend=impl)
    # model_kwargs["naconmax"] = 800
    model_kwargs["naconmax"] = 10_000
    model = Model(MODEL_PATH, **model_kwargs)
    sensors = build_walker_sensors(model.mujoco_model)
    values = model.model, *sensors, target_velocity, target_height
    walker_args = WalkerArgs(*values)
    running = partial(running_cost, walker_args)
    return Task(running), model


def build_walker_sensors(mujoco_model):
    position_sensor = get_sensor_id(mujoco_model, "torso_position")
    velocity_sensor = get_sensor_id(mujoco_model, "torso_subtreelinvel")
    zaxis_sensor = get_sensor_id(mujoco_model, "torso_zaxis")
    return position_sensor, velocity_sensor, zaxis_sensor


def get_torso_height(args, state):
    sensor_address = args.model.sensor_adr[args.torso_position_sensor]
    return state.sensordata[sensor_address + 2]


def get_torso_velocity(args, state):
    sensor_address = args.model.sensor_adr[args.torso_velocity_sensor]
    return state.sensordata[sensor_address]


def get_torso_deviation(args, state):
    sensor_address = args.model.sensor_adr[args.torso_zaxis_sensor]
    return state.sensordata[sensor_address + 2] - 1.0


def running_cost(args, state, control):
    state_value = state_cost(args, state)
    control_cost = 0.5 * jp.sum(jp.square(control))
    return state_value + 0.1 * control_cost


def state_cost(args, state):
    height = get_torso_height(args, state) - args.target_height
    height_cost = 0.5 * jp.square(height)
    orient_cost = 0.5 * jp.square(get_torso_deviation(args, state))
    velocity = get_torso_velocity(args, state) - args.target_velocity
    velocity_cost = 0.5 * jp.square(velocity)
    return 10.0 * height_cost + 3.0 * orient_cost + velocity_cost


def get_sensor_id(mujoco_model, name):
    return mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", action="store_true")
    parser.add_argument("--target-velocity", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=1.2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-traces", type=int, default=5)
    parser.add_argument("--hide-traces", action="store_true")
    return parser.parse_args()


def build_controller(task, model, iterations):
    # sampler_args = model, 10, 3, 0.8, 0.5, interpolate_cubic
    sampler_args = model, 1024, 3, 0.8, 0.5, interpolate_cubic
    sampler = KnotSampler(*sampler_args)
    return PredictiveSampler(task, model, sampler, iterations)


def build_simulation_model(model):
    mujoco_model = deepcopy(model.mujoco_model)
    mujoco_model.opt.timestep = 0.0025
    mujoco_model.opt.iterations = 100
    mujoco_model.opt.ls_iterations = 50
    return mujoco_model


def main():
    args = parse_args()
    target_velocity = args.target_velocity
    task_args = "warp" if args.warp else "jax", target_velocity
    task, model = Walker(*task_args, target_height=args.target_height)
    controller = build_controller(task, model, args.iterations)
    mujoco_model = build_simulation_model(model)
    mj_data = mujoco.MjData(mujoco_model)
    parameters = controller.sampler.initialize()
    viewer_args = controller, mujoco_model, mj_data, parameters
    viewer_kwargs = dict(frequency=50, fixed_camera_id=0)
    viewer_kwargs["show_traces"] = not args.hide_traces
    viewer_kwargs["max_traces"] = args.max_traces
    run_interactive(*viewer_args, **viewer_kwargs)


if __name__ == "__main__":
    main()
