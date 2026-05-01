import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import jax.numpy as jp
import mujoco

try:
    from .controller import PredictiveSampling
    from .spline import interpolate_zero
    from .structures import WalkerArgs
    from .structures import WalkerTask
    from .task import build_task
    from .viewer import run_interactive
except ImportError:
    from controller import PredictiveSampling
    from spline import interpolate_zero
    from structures import WalkerArgs
    from structures import WalkerTask
    from task import build_task
    from viewer import run_interactive


MODEL_PATH = Path(__file__).parent / "models" / "walker" / "scene.xml"


def Walker(impl="jax"):
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    base = build_task(mj_model, trace_sites=("torso_site",), impl=impl)
    sensors = build_walker_sensors(mj_model)
    walker_args = WalkerArgs(base.model, *sensors, 1.5, 1.2)
    functions = build_walker_functions(base, walker_args)
    base = base._replace(**functions)
    return WalkerTask(*base, *sensors, 1.5, 1.2)


def build_walker_sensors(mj_model):
    position_sensor = get_sensor_id(mj_model, "torso_position")
    velocity_sensor = get_sensor_id(mj_model, "torso_subtreelinvel")
    zaxis_sensor = get_sensor_id(mj_model, "torso_zaxis")
    return position_sensor, velocity_sensor, zaxis_sensor


def build_walker_functions(task, walker_args):
    functions = {}
    functions["running_cost"] = partial(running_cost, walker_args)
    functions["terminal_cost"] = partial(terminal_cost, walker_args)
    functions["make_data"] = partial(make_walker_data, task.make_data)
    return functions


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
    state_cost = terminal_cost(args, state)
    control_cost = jp.sum(jp.square(control))
    return state_cost + 0.1 * control_cost


def terminal_cost(args, state):
    height = get_torso_height(args, state) - args.target_height
    height_cost = jp.square(height)
    orient_cost = jp.square(get_torso_deviation(args, state))
    velocity = get_torso_velocity(args, state) - args.target_velocity
    velocity_cost = jp.square(velocity)
    return 10.0 * height_cost + 3.0 * orient_cost + velocity_cost


def make_walker_data(make_data):
    return make_data(naconmax=800)


def get_sensor_id(mj_model, name):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", action="store_true")
    return parser.parse_args()


def build_controller(task):
    # args = task, 128, 0.5, 0.6, interpolate_zero, 5, 1
    args = task, 256, 0.5, 1.0, interpolate_zero, 5, 1
    return PredictiveSampling(*args)


def build_simulation_model(task):
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 50
    return mj_model


def main():
    args = parse_args()
    task = Walker(impl="warp" if args.warp else "jax")
    controller = build_controller(task)
    mj_model = build_simulation_model(task)
    mj_data = mujoco.MjData(mj_model)
    viewer_args = controller, mj_model, mj_data
    viewer_kwargs = dict(frequency=50, fixed_camera_id=0, show_traces=False)
    viewer_kwargs["max_traces"] = 1
    run_interactive(*viewer_args, **viewer_kwargs)


if __name__ == "__main__":
    main()
