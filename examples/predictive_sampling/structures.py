from collections import namedtuple

Parameters = namedtuple("Parameters", ["knot_times", "mean"])  # fmt: skip
Trajectory = namedtuple("Trajectory", ["controls", "knots", "costs", "trace_sites"])  # fmt: skip
Task = namedtuple("Task", ["running_cost", "terminal_cost"])  # fmt: skip
Sampler = namedtuple("Sampler", ["plan_horizon", "interpolate", "initialize", "warm_start", "sample", "update"])  # fmt: skip
Controller = namedtuple("Controller", ["task", "model", "sampler", "iterations", "num_control_steps", "optimize"])  # fmt: skip

dynamics_fields = ["mujoco_model", "model", "u_min", "u_max"]
dynamics_fields += ["num_actuators", "time_delta", "trace_site_ids"]
dynamics_fields += ["make_state", "get_trace_positions"]
Dynamics = namedtuple("Dynamics", dynamics_fields)
