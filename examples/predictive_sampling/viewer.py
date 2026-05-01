import os
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np


def _print_plan(controller):
    first = f"Planning with {controller.num_control_steps} steps"
    second = f"over a {controller.plan_horizon} second horizon"
    print(f"{first} {second} with {controller.num_knots} knots.")


def _compute_replan_args(model, frequency):
    replan_time = 1.0 / frequency
    num_steps = max(int(replan_time / model.opt.timestep), 1)
    step_time = num_steps * model.opt.timestep
    first = f"Planning at {1.0 / step_time} Hz"
    print(f"{first}, simulating at {1.0 / model.opt.timestep} Hz")
    return num_steps, step_time, 1.0 / step_time


def _build_rollout_data(controller, mj_data):
    rollout_data = controller.task.make_data()
    data_kwargs = dict(qpos=mj_data.qpos, qvel=mj_data.qvel)
    data_kwargs["mocap_pos"] = mj_data.mocap_pos
    data_kwargs["mocap_quat"] = mj_data.mocap_quat
    return rollout_data.replace(**data_kwargs)


def _update_rollout_data(rollout_data, mj_data):
    data_kwargs = dict(qpos=jp.array(mj_data.qpos), qvel=jp.array(mj_data.qvel))
    data_kwargs["mocap_pos"] = jp.array(mj_data.mocap_pos)
    data_kwargs["mocap_quat"] = jp.array(mj_data.mocap_quat)
    data_kwargs["time"] = mj_data.time
    return rollout_data.replace(**data_kwargs)


def _compute_controls(interpolate, model, num_steps, policy, current_time):
    sim_time = model.opt.timestep
    query_times = jp.arange(num_steps) * sim_time + current_time
    knot_times = policy.knot_times
    knots = policy.mean[None, ...]
    return np.asarray(interpolate(query_times, knot_times, knots))[0]


def _warm_up(optimize, interpolate, rollout_data, policy, key, model, steps):
    print("Jitting the controller...")
    start_time = time.time()
    policy, rollouts, key = optimize(rollout_data, policy, key)
    policy, rollouts, key = optimize(rollout_data, policy, key)
    _compute_controls(interpolate, model, steps, policy, 0.0)
    _compute_controls(interpolate, model, steps, policy, 0.0)
    print(f"Time to jit: {time.time() - start_time:.3f} seconds")
    return policy, rollouts, key


def _build_reference(reference, model):
    if reference is None:
        return None
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


def _build_video(model, actual_frequency, record_video):
    if not record_video:
        return None, None
    from hydrax import ROOT
    from hydrax.utils.video import VideoRecorder

    recorder_kwargs = dict(output_dir=os.path.join(ROOT, "recordings"))
    recorder_kwargs["width"] = 720
    recorder_kwargs["height"] = 480
    recorder_kwargs["fps"] = actual_frequency
    recorder = VideoRecorder(**recorder_kwargs)
    model.vis.global_.offwidth = recorder_kwargs["width"]
    model.vis.global_.offheight = recorder_kwargs["height"]
    if not recorder.start():
        return None, None
    renderer = mujoco.Renderer(model, height=480, width=720)
    return recorder, renderer


def _set_camera(viewer, fixed_camera_id):
    if fixed_camera_id is None:
        return
    viewer.cam.fixedcamid = fixed_camera_id
    viewer.cam.type = 2


def _init_trace_geom(geom, color):
    args = geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3)
    kwargs = dict(pos=np.zeros(3), mat=np.eye(3).flatten())
    kwargs["rgba"] = np.array(color)
    mujoco.mjv_initGeom(*args, **kwargs)


def _init_traces(viewer, controller, num_traces, color):
    num_sites = len(controller.task.trace_site_ids)
    num_geoms = num_sites * num_traces * controller.num_control_steps
    for geom_index in range(num_geoms):
        _init_trace_geom(viewer.user_scn.geoms[geom_index], color)
        viewer.user_scn.ngeom += 1
    return num_sites


def _update_traces(viewer, rollouts, num_sites, num_traces, num_steps, width):
    geom_index = 0
    for site_index in range(num_sites):
        for trace_index in range(num_traces):
            trace_points = rollouts.trace_sites[trace_index, :, site_index]
            for step_index in range(num_steps):
                geom = viewer.user_scn.geoms[geom_index]
                args = geom, mujoco.mjtGeom.mjGEOM_LINE
                points = trace_points[step_index:step_index + 2]
                mujoco.mjv_connector(*args, width, points[0], points[1])
                geom_index += 1


def _step_simulation(model, mj_data, viewer, controls, recorder, renderer):
    for control in controls:
        mj_data.ctrl[:] = np.array(control)
        mujoco.mj_step(model, mj_data)
        viewer.sync()
        if recorder is None or not recorder.is_recording:
            continue
        renderer.update_scene(mj_data, viewer.cam)
        frame = renderer.render()
        recorder.add_frame(frame.tobytes())


def _wait_for_step(step_time, loop_start, plan_time):
    elapsed = time.time() - loop_start
    if elapsed < step_time:
        time.sleep(step_time - elapsed)
    realtime_rate = step_time / (time.time() - loop_start)
    message = f"Realtime rate: {realtime_rate:.2f}, plan time: {plan_time:.4f}s"
    print(message, end="\r")


def build_viewer_config(config):
    defaults = {}
    defaults["initial_knots"] = None
    defaults["fixed_camera_id"] = None
    defaults["show_traces"] = True
    defaults["max_traces"] = 5
    defaults["trace_width"] = 5.0
    defaults["trace_color"] = (1.0, 1.0, 1.0, 0.1)
    defaults["reference"] = None
    defaults["reference_fps"] = 30.0
    defaults["record_video"] = False
    defaults.update(config)
    return defaults


def run_interactive(controller, mj_model, mj_data, frequency, **config):
    config = build_viewer_config(config)
    initial_knots = config["initial_knots"]
    fixed_camera_id = config["fixed_camera_id"]
    show_traces = config["show_traces"]
    max_traces = config["max_traces"]
    trace_width = config["trace_width"]
    trace_color = config["trace_color"]
    reference = config["reference"]
    reference_fps = config["reference_fps"]
    record_video = config["record_video"]
    _print_plan(controller)
    replan_args = _compute_replan_args(mj_model, frequency)
    num_steps, step_time, actual_frequency = replan_args
    rollout_data = _build_rollout_data(controller, mj_data)
    key = jax.random.key(0)
    policy = controller.initialize_parameters(initial_knots=initial_knots)
    optimize = jax.jit(controller.optimize)
    interpolate = jax.jit(controller.interpolate)
    warm_up_args = optimize, interpolate, rollout_data, policy, key
    policy, rollouts, key = _warm_up(*warm_up_args, mj_model, num_steps)
    num_traces = min(rollouts.controls.shape[1], max_traces)
    reference_view = _build_reference(reference, mj_model)
    recorder, renderer = _build_video(mj_model, actual_frequency, record_video)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        _set_camera(viewer, fixed_camera_id)
        num_sites = 0
        if show_traces:
            trace_args = viewer, controller, num_traces, trace_color
            num_sites = _init_traces(*trace_args)
        if reference_view is not None:
            _add_reference(mj_model, reference_view, viewer)
        while viewer.is_running():
            loop_start = time.time()
            rollout_data = _update_rollout_data(rollout_data, mj_data)
            plan_start = time.time()
            policy, rollouts, key = optimize(rollout_data, policy, key)
            plan_time = time.time() - plan_start
            if show_traces:
                trace_args = viewer, rollouts, num_sites, num_traces
                num_trace_steps = controller.num_control_steps
                _update_traces(*trace_args, num_trace_steps, trace_width)
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
        recorder.stop()
