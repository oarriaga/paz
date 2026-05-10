import os
import time
from collections import namedtuple

import jax
import jax.numpy as jp
import jax.random as jr
import mujoco
import mujoco.viewer
import numpy as np

from video import start_recording, add_frame, stop_recording

TraceView = namedtuple("TraceView", "num_sites num_samples best_start")


def _print_plan(controller, policy):
    sampler = controller.sampler
    num_knots = policy.mean.shape[0]
    first = f"Planning with {controller.num_control_steps} steps"
    second = f"and {num_knots} knots over a {sampler.plan_horizon}s horizon"
    print(f"{first} {second}")


def _compute_replan_args(model, frequency):
    replan_time = 1.0 / frequency
    num_steps = max(int(replan_time / model.opt.timestep), 1)
    step_time = num_steps * model.opt.timestep
    first = f"Replanning at {1.0 / step_time} Hz"
    print(f"{first}, simulating at {1.0 / model.opt.timestep} Hz")
    return num_steps, step_time, 1.0 / step_time


def _build_state(controller, mj_data):
    state = controller.model.make_state()
    data_kwargs = dict(qpos=mj_data.qpos, qvel=mj_data.qvel)
    data_kwargs["mocap_pos"] = mj_data.mocap_pos
    data_kwargs["mocap_quat"] = mj_data.mocap_quat
    return state.replace(**data_kwargs)


def _update_state(state, mj_data):
    data_kwargs = dict(qpos=jp.array(mj_data.qpos), qvel=jp.array(mj_data.qvel))
    data_kwargs["mocap_pos"] = jp.array(mj_data.mocap_pos)
    data_kwargs["mocap_quat"] = jp.array(mj_data.mocap_quat)
    data_kwargs["time"] = mj_data.time
    return state.replace(**data_kwargs)


def _compute_controls(interpolate, model, num_steps, policy, current_time):
    sim_time = model.opt.timestep
    query_times = jp.arange(num_steps) * sim_time + current_time
    knot_times = policy.knot_times
    knots = policy.mean[None, ...]
    return np.asarray(interpolate(query_times, knot_times, knots))[0]


def _warm_up(optimize, interpolate, state, policy, key, model, steps):
    print("Jitting the controller...")
    start_time = time.time()
    key, subkey = jr.split(key)
    policy, rollouts = optimize(subkey, state, policy)
    _compute_controls(interpolate, model, steps, policy, 0.0)
    print(f"Time to jit: {time.time() - start_time:.3f} seconds")
    return policy, rollouts, key


def _build_reference(reference, model):
    reference_data = mujoco.MjData(model)
    assert reference.shape[1] == model.nq
    reference_data.qpos[:] = reference[0]
    mujoco.mj_forward(model, reference_data)
    visual_options = mujoco.MjvOption()
    visual_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    perturbation = mujoco.MjvPerturb()
    category_mask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    return reference_data, visual_options, perturbation, category_mask


def _add_reference(model, view, viewer):
    reference_data, visual_options, perturbation, category_mask = view
    args = model, reference_data, visual_options, perturbation
    mujoco.mjv_addGeoms(*args, category_mask, viewer.user_scn)


def _update_reference(model, mj_data, reference, reference_fps, view, viewer):
    reference_time = mj_data.time * reference_fps
    reference_index = min(int(reference_time), reference.shape[0] - 1)
    reference_data, visual_options, perturbation, category_mask = view
    reference_data.qpos[:] = reference[reference_index]
    mujoco.mj_forward(model, reference_data)
    args = model, reference_data, visual_options, perturbation
    mujoco.mjv_updateScene(*args, viewer.cam, category_mask, viewer.user_scn)


def _build_video(model, replan_frequency):
    output_dir = os.path.join(os.path.dirname(__file__), "recordings")
    recorder = start_recording(output_dir, 720, 480, replan_frequency)
    if recorder is None:
        return None, None
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model, height=480, width=720)
    return recorder, renderer


def _set_camera(viewer, fixed_camera_id):
    if fixed_camera_id is None:
        return
    viewer.cam.fixedcamid = fixed_camera_id
    viewer.cam.type = 2


def _init_trace_geom(geom, geom_type, color):
    args = geom, geom_type, np.zeros(3)
    kwargs = dict(pos=np.zeros(3), mat=np.eye(3).flatten())
    kwargs["rgba"] = np.array(color)
    mujoco.mjv_initGeom(*args, **kwargs)


def _init_traces(viewer, controller, num_samples, sample_color, best_color):
    num_sites = len(controller.model.trace_site_ids)
    num_segments = controller.num_control_steps
    num_sample_geoms = num_sites * num_samples * num_segments
    num_best_geoms = num_sites * num_segments
    for geom_index in range(num_sample_geoms):
        geom = viewer.user_scn.geoms[geom_index]
        _init_trace_geom(geom, mujoco.mjtGeom.mjGEOM_LINE, sample_color)
        viewer.user_scn.ngeom += 1
    for geom_index in range(num_best_geoms):
        scene_index = num_sample_geoms + geom_index
        geom = viewer.user_scn.geoms[scene_index]
        _init_trace_geom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, best_color)
        viewer.user_scn.ngeom += 1
    return TraceView(num_sites, num_samples, num_sample_geoms)


def _compute_best_rollout(rollouts):
    costs = np.mean(np.asarray(rollouts.costs), axis=1)
    return int(np.argmin(costs))


def _select_sample_rollouts(num_rollouts, best_index, num_samples):
    rollouts = [index for index in range(num_rollouts) if index != best_index]
    return rollouts[:num_samples]


def _update_trace_path(viewer, rollouts, geom_index, args):
    rollout_index, site_index, num_steps, width, geom_type = args
    trace_sites = rollouts.trace_sites
    trace_points = np.asarray(trace_sites[rollout_index, :, site_index])
    for step_index in range(num_steps):
        geom = viewer.user_scn.geoms[geom_index]
        points = trace_points[step_index : step_index + 2]
        mujoco.mjv_connector(geom, geom_type, width, points[0], points[1])
        geom_index += 1
    return geom_index


def _update_sample_traces(viewer, rollouts, view, best_index, num_steps, width):
    num_rollouts = rollouts.trace_sites.shape[0]
    args = num_rollouts, best_index, view.num_samples
    indices = _select_sample_rollouts(*args)
    geom_index = 0
    for site_index in range(view.num_sites):
        for rollout_index in indices:
            geom_type = mujoco.mjtGeom.mjGEOM_LINE
            args = rollout_index, site_index, num_steps, width, geom_type
            geom_index = _update_trace_path(viewer, rollouts, geom_index, args)


def _update_best_trace(viewer, rollouts, view, best_index, num_steps, width):
    geom_index = view.best_start
    geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
    for site_index in range(view.num_sites):
        args = best_index, site_index, num_steps, width, geom_type
        geom_index = _update_trace_path(viewer, rollouts, geom_index, args)


def _update_traces(viewer, rollouts, view, num_steps, sample_width, best_width):
    best_index = _compute_best_rollout(rollouts)
    args = viewer, rollouts, view, best_index, num_steps, sample_width
    _update_sample_traces(*args)
    args = viewer, rollouts, view, best_index, num_steps, best_width
    _update_best_trace(*args)


def _step_simulation(model, mj_data, viewer, controls, recorder, renderer):
    for control in controls:
        mj_data.ctrl[:] = np.array(control)
        mujoco.mj_step(model, mj_data)
        viewer.sync()
        if recorder is None:
            continue
        renderer.update_scene(mj_data, viewer.cam)
        frame = renderer.render()
        add_frame(recorder, frame.tobytes())


def _wait_for_step(step_time, loop_start, plan_time):
    elapsed = time.time() - loop_start
    if elapsed < step_time:
        time.sleep(step_time - elapsed)
    realtime_rate = step_time / (time.time() - loop_start)
    message = f"Realtime rate: {realtime_rate:.2f}, plan time: {plan_time:.4f}s"
    print(message, end="\r")


def run_interactive(*args, **kwargs):
    controller, mj_model, mj_data, policy = args[:4]
    frequency = _pop_frequency(args, kwargs)
    fixed_camera_id = kwargs.pop("fixed_camera_id", None)
    show_traces = kwargs.pop("show_traces", True)
    max_traces = kwargs.pop("max_traces", 5)
    trace_width = kwargs.pop("trace_width", 3.0)
    trace_color = kwargs.pop("trace_color", (1.0, 1.0, 1.0, 0.15))
    best_trace_width = kwargs.pop("best_trace_width", 0.015)
    best_color = kwargs.pop("best_trace_color", (1.0, 0.0, 1.0, 1.0))
    reference = kwargs.pop("reference", None)
    reference_fps = kwargs.pop("reference_fps", 30.0)
    record_video = kwargs.pop("record_video", False)
    if kwargs:
        raise ValueError(f"Unknown viewer options: {tuple(kwargs)}")
    _print_plan(controller, policy)
    replan_args = _compute_replan_args(mj_model, frequency)
    num_steps, step_time, replan_frequency = replan_args
    state = _build_state(controller, mj_data)
    key = jax.random.key(0)
    optimize = jax.jit(controller.optimize)
    interpolate = jax.jit(controller.sampler.interpolate)
    warm_up_args = optimize, interpolate, state, policy, key
    policy, rollouts, key = _warm_up(*warm_up_args, mj_model, num_steps)
    num_rollouts = rollouts.controls.shape[0]
    num_samples = min(max(num_rollouts - 1, 0), max(max_traces - 1, 0))
    reference_view = None
    if reference is not None:
        reference_view = _build_reference(reference, mj_model)
    recorder, renderer = None, None
    if record_video:
        recorder, renderer = _build_video(mj_model, replan_frequency)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        _set_camera(viewer, fixed_camera_id)
        trace_view = None
        if show_traces:
            trace_args = viewer, controller, num_samples
            trace_colors = trace_color, best_color
            trace_view = _init_traces(*trace_args, *trace_colors)
        if reference_view is not None:
            _add_reference(mj_model, reference_view, viewer)
        while viewer.is_running():
            loop_start = time.time()
            state = _update_state(state, mj_data)
            plan_start = time.time()
            key, subkey = jr.split(key)
            policy, rollouts = optimize(subkey, state, policy)
            plan_time = time.time() - plan_start
            if show_traces:
                num_trace_steps = controller.num_control_steps
                trace_args = viewer, rollouts, trace_view, num_trace_steps
                _update_traces(*trace_args, trace_width, best_trace_width)
            if reference_view is not None:
                ref_args = mj_model, mj_data, reference, reference_fps
                _update_reference(*ref_args, reference_view, viewer)
            control_time = mj_data.time
            control_args = interpolate, mj_model, num_steps, policy
            controls = _compute_controls(*control_args, control_time)
            sim_args = mj_model, mj_data, viewer, controls
            _step_simulation(*sim_args, recorder, renderer)
            _wait_for_step(step_time, loop_start, plan_time)
    print()
    if recorder is not None:
        stop_recording(recorder)


def _pop_frequency(args, kwargs):
    if len(args) > 4:
        return args[4]
    return kwargs.pop("frequency")
