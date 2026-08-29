import keyword
import re
from collections import namedtuple

import jax.numpy as jp
import mujoco

Robot = namedtuple("Robot", "mjspec, configure, bodies, joints, frames, sensors, keyframes, num_actuators, control_limits, joint_limits, time_delta, contact_bodies")  # fmt: skip
Body = namedtuple("Body", "name, arg")
Frame = namedtuple("Frame", "name, arg")
Joint = namedtuple("Joint", "name, arg, qpos_address, dof_address")
Sensor = namedtuple("Sensor", "name, arg, address, dimension")
Keyframe = namedtuple("Keyframe", "name, arg, qpos, ctrl")


def build(mjspec, configure, contact_suffix):
    model = configure(mjspec.compile())
    bodies = build_bodies(model)
    joints = build_joints(model)
    frames = build_frames(model)
    sensors = build_sensors(model)
    keyframes = build_keyframes(model)
    control_limits = compute_control_limits(model)
    joint_limits = compute_joint_limits(model)
    contact_bodies = select_args(bodies, contact_suffix)
    robot_args = mjspec, configure, bodies, joints, frames, sensors, keyframes, model.nu, control_limits, joint_limits, model.opt.timestep, contact_bodies  # fmt: skip
    return Robot(*robot_args)


def build_bodies(model):
    args = "Bodies", model, mujoco.mjtObj.mjOBJ_BODY, model.nbody, build_body
    return build_lookup(*args)


def build_joints(model):
    object_type = mujoco.mjtObj.mjOBJ_JOINT
    args = "Joints", model, object_type, model.njnt, build_joint
    return build_lookup(*args)


def build_frames(model):
    args = "Frames", model, mujoco.mjtObj.mjOBJ_SITE, model.nsite, build_frame
    return build_lookup(*args)


def build_sensors(model):
    object_type = mujoco.mjtObj.mjOBJ_SENSOR
    args = "Sensors", model, object_type, model.nsensor, build_sensor
    return build_lookup(*args)


def build_keyframes(model):
    object_type = mujoco.mjtObj.mjOBJ_KEY
    args = "Keyframes", model, object_type, model.nkey, build_keyframe
    return build_lookup(*args)


def build_lookup(type_name, model, object_type, num_objects, build_object):
    entries = []
    for arg in range(num_objects):
        name = mujoco.mj_id2name(model, object_type, arg)
        if name is not None:
            entries.append((name, build_object(model, name, arg)))
    return build_namedtuple(type_name, entries)


def build_body(model, name, arg):
    return Body(name, arg)


def build_frame(model, name, arg):
    return Frame(name, arg)


def build_joint(model, name, arg):
    qpos_address = int(model.jnt_qposadr[arg])
    dof_address = int(model.jnt_dofadr[arg])
    return Joint(name, arg, qpos_address, dof_address)


def build_sensor(model, name, arg):
    address = int(model.sensor_adr[arg])
    dimension = int(model.sensor_dim[arg])
    return Sensor(name, arg, address, dimension)


def build_keyframe(model, name, arg):
    qpos = jp.array(model.key_qpos[arg])
    ctrl = jp.array(model.key_ctrl[arg])
    return Keyframe(name, arg, qpos, ctrl)


def compute_joint_limits(model):
    args = []
    for arg in range(model.njnt):
        if model.jnt_type[arg] != mujoco.mjtJoint.mjJNT_FREE:
            args.append(arg)
    ranges = model.jnt_range[args]
    return jp.asarray(ranges[:, 0]), jp.asarray(ranges[:, 1])


def compute_control_limits(model):
    limited = model.actuator_ctrllimited
    lower = jp.where(limited, model.actuator_ctrlrange[:, 0], -jp.inf)
    upper = jp.where(limited, model.actuator_ctrlrange[:, 1], jp.inf)
    return lower, upper


def select_keyword_args(objects, keywords):
    args = []
    for object_ in objects:
        if any(keyword in object_.name for keyword in keywords):
            args.append(object_.arg)
    return jp.array(args)


def read_sensor_addresses(sensors, names):
    addresses = []
    for name in names:
        sensor = getattr(sensors, name)
        addresses.extend(range(sensor.address, sensor.address + sensor.dimension))  # fmt: skip
    return jp.array(addresses)


def select_args(objects, suffix):
    args = []
    for object_ in objects:
        if object_.name.endswith(suffix):
            args.append(object_.arg)
    return jp.array(args)


def reject_args(objects, suffix):
    args = []
    for object_ in objects:
        if not object_.name.endswith(suffix):
            args.append(object_.arg)
    return jp.array(args)


def reject_keyword_args(objects, keyword):
    args = []
    for object_ in objects:
        if keyword not in object_.name:
            args.append(object_.arg)
    return jp.array(args)


def build_namedtuple(type_name, entries):
    fields, values, used = [], [], {}
    for name, value in entries:
        fields.append(build_unique_field(name, used))
        values.append(value)
    return namedtuple(type_name, fields)(*values)


def build_unique_field(name, used):
    field = build_field(name)
    count = used.get(field, 0)
    used[field] = count + 1
    if count == 0:
        unique_field = field
    else:
        unique_field = f"{field}_{count}"
    return unique_field


def build_field(name):
    field = re.sub(r"\W+", "_", name).strip("_")
    if field[0].isdigit():
        field = f"object_{field}"
    if keyword.iskeyword(field):
        field = f"{field}_"
    return field
