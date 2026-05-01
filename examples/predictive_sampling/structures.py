from collections import namedtuple


parameter_fields = "knot_times mean"
Parameters = namedtuple("Parameters", parameter_fields)

trajectory_fields = "controls knots costs trace_sites"
Trajectory = namedtuple("Trajectory", trajectory_fields)

task_fields = "mj_model model u_min u_max time_delta trace_site_ids"
task_fields += " running_cost terminal_cost make_data get_trace_sites"
Task = namedtuple("Task", task_fields)

controller_arg_fields = "num_samples noise_level plan_horizon num_knots"
controller_arg_fields += " iterations num_control_steps interpolate"
ControllerArgs = namedtuple("ControllerArgs", controller_arg_fields)

controller_fields = "task " + controller_arg_fields
controller_fields += " initialize_parameters optimize"
Controller = namedtuple("PredictiveSampling", controller_fields)

walker_arg_fields = "model torso_position_sensor torso_velocity_sensor"
walker_arg_fields += " torso_zaxis_sensor target_velocity target_height"
WalkerArgs = namedtuple("WalkerArgs", walker_arg_fields)

walker_fields = task_fields + " torso_position_sensor"
walker_fields += " torso_velocity_sensor torso_zaxis_sensor"
walker_fields += " target_velocity target_height"
WalkerTask = namedtuple("Walker", walker_fields)
