"""MuJoCo support code for the PAZ SONIC demonstration."""

from collections import deque
from collections import namedtuple
from pathlib import Path
import re

import mujoco
import numpy as np

from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.layout import find_encoder_span


MotionClip = namedtuple(
    "MotionClip",
    "name joint_pos joint_vel body_pos body_quat body_indices "
    "smpl_joints num_frames",
)
HistoryEntry = namedtuple(
    "HistoryEntry",
    "base_quat base_ang_vel body_q body_dq last_action",
)
HeadingState = namedtuple(
    "HeadingState",
    "init_base_quat init_ref_root_quat delta_heading",
)
JointAddresses = namedtuple(
    "JointAddresses",
    "body_qpos body_dof body_actuator hand_qpos hand_dof hand_actuator",
)

NUM_JOINTS = 29
ROOT_QPOS_OFFSET = 7
ROOT_QVEL_OFFSET = 6
HISTORY_FRAMES = 10

SCENE_FROM_POLICY = np.asarray(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int32,
)
POLICY_FROM_SCENE = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int32,
)
LOWER_BODY_POLICY = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int32)
WRIST_POLICY = np.asarray([23, 24, 25, 26, 27, 28], dtype=np.int32)
VR_BODY_IDS = np.asarray([28, 29, 9], dtype=np.int32)
VR_OFFSETS = np.asarray(
    [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
    dtype=np.float32,
)

DEFAULT_ANGLES = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0, 0.2, 0.2, 0.0,
        0.6, 0.0, 0.0, 0.0, 0.2, -0.2,
        0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

NATURAL_FREQ = 10.0 * 2.0 * np.pi
DAMPING_RATIO = 2.0
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = (
    2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ)
DAMPING_7520_22 = (
    2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ)
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0

ACTION_SCALE = np.asarray(
    [
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
    ],
    dtype=np.float32,
)
KPS = np.asarray(
    [
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
        STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
        STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_7520_14, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010,
        STIFFNESS_4010, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_4010, STIFFNESS_4010,
    ],
    dtype=np.float32,
)
KDS = np.asarray(
    [
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
        DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
        DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_7520_14, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_4010,
        DAMPING_4010, DAMPING_5020, DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_4010, DAMPING_4010,
    ],
    dtype=np.float32,
)
EFFORT_LIMITS = np.asarray(
    [
        88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50,
        88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25,
        25, 5, 5,
    ],
    dtype=np.float32,
)
HAND_EFFORT_LIMITS = np.asarray(
    [2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7] * 2,
    dtype=np.float32,
)


SUPPORT_POINT = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
SUPPORT_KP_POS = 10000.0
SUPPORT_KD_POS = 1000.0
SUPPORT_KP_ANG = 1000.0
SUPPORT_KD_ANG = 10.0


def load_motion_set(motion_dir):
    motion_dir = Path(motion_dir)
    clip_paths = sorted(
        path for path in motion_dir.iterdir()
        if path.is_dir() and (path / "joint_pos.csv").exists())
    clips = tuple(load_motion_clip(path) for path in clip_paths)
    if not clips:
        raise ValueError(f"No SONIC motion folders found in {motion_dir}")
    return clips


def load_motion_clip(clip_path):
    clip_path = Path(clip_path)
    joint_pos = load_csv_array(clip_path / "joint_pos.csv")
    joint_vel = load_csv_array(clip_path / "joint_vel.csv")
    body_pos = load_csv_array(clip_path / "body_pos.csv")
    body_quat = load_csv_array(clip_path / "body_quat.csv")
    num_frames = joint_pos.shape[0]
    body_pos = body_pos.reshape(num_frames, -1, 3)
    body_quat = body_quat.reshape(num_frames, -1, 4)
    body_indices = load_body_indices(clip_path / "metadata.txt")
    smpl_path = clip_path / "smpl_joints.csv"
    smpl_joints = None
    if smpl_path.exists():
        smpl_joints = load_csv_array(smpl_path).reshape(num_frames, 24, 3)
    arrays = joint_vel, body_pos, body_quat
    if any(array.shape[0] != num_frames for array in arrays):
        raise ValueError(f"Inconsistent frame counts in {clip_path}")
    if body_pos.shape[1] != body_indices.size:
        raise ValueError(f"Body index count does not match {clip_path}")
    args = (
        clip_path.name, joint_pos, joint_vel, body_pos, body_quat,
        body_indices, smpl_joints, num_frames,
    )
    return MotionClip(*args)


def load_csv_array(path):
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return values


def load_body_indices(metadata_path):
    text = Path(metadata_path).read_text(encoding="utf-8")
    match = re.search(r"Body part indexes:\s*\[([^]]+)\]", text, re.S)
    if match is None:
        raise ValueError(f"No body indexes found in {metadata_path}")
    return np.fromstring(match.group(1), sep=" ", dtype=np.int32)


def find_mode(layout, mode_name):
    for mode in layout.encoder_modes:
        if mode.name == mode_name:
            return mode
    raise KeyError(f"Unknown SONIC mode: {mode_name}")


def check_mode_available(clip, mode_name):
    if mode_name == "smpl" and clip.smpl_joints is None:
        return False, "clip has no smpl_joints.csv"
    if mode_name == "teleop":
        missing = set(VR_BODY_IDS) - set(clip.body_indices)
        if missing:
            return False, f"clip is missing VR body IDs {sorted(missing)}"
    return True, ""


def build_encoder_obs(
    layout, mode_name, clip, frame, play, base_quat, heading_state
):
    available, reason = check_mode_available(clip, mode_name)
    if not available:
        raise ValueError(f"Cannot use {mode_name} mode: {reason}")
    if mode_name == "g1":
        return build_g1_encoder_obs(
            layout, clip, frame, play, base_quat, heading_state)
    if mode_name == "teleop":
        return build_teleop_encoder_obs(
            layout, clip, frame, play, base_quat, heading_state)
    if mode_name == "smpl":
        return build_smpl_encoder_obs(
            layout, clip, frame, play, base_quat, heading_state)
    raise KeyError(f"Unsupported SONIC mode: {mode_name}")


def build_empty_encoder_obs(layout, mode_name):
    obs = np.zeros((compute_encoder_input_dim(layout),), dtype=np.float32)
    mode = find_mode(layout, mode_name)
    span = find_encoder_span(layout, layout.mode_observation_name)
    obs[span.start] = np.float32(mode.mode_id)
    return obs


def build_g1_encoder_obs(
    layout, clip, frame, play, base_quat, heading_state
):
    obs = build_empty_encoder_obs(layout, "g1")
    write_span(
        obs, layout, "motion_joint_positions_10frame_step5",
        gather_motion_values(clip.joint_pos, clip, frame, play, 10, 5),
    )
    velocities = gather_motion_values(
        clip.joint_vel, clip, frame, play, 10, 5)
    if not play:
        velocities[:] = 0.0
    write_span(
        obs, layout, "motion_joint_velocities_10frame_step5",
        velocities,
    )
    orientations = compute_anchor_orientations(
        clip, frame, play, 10, 5, base_quat, heading_state)
    write_span(
        obs, layout, "motion_anchor_orientation_10frame_step5",
        orientations,
    )
    return obs[None, :]


def build_teleop_encoder_obs(
    layout, clip, frame, play, base_quat, heading_state
):
    obs = build_empty_encoder_obs(layout, "teleop")
    positions = gather_motion_values(
        clip.joint_pos[:, LOWER_BODY_POLICY], clip, frame, play, 10, 5)
    velocities = gather_motion_values(
        clip.joint_vel[:, LOWER_BODY_POLICY], clip, frame, play, 10, 5)
    if not play:
        velocities[:] = 0.0
    vr_positions, vr_orientations = compute_vr_targets(clip, frame)
    anchor = compute_anchor_orientations(
        clip, frame, play, 1, 1, base_quat, heading_state)
    write_span(
        obs, layout, "motion_joint_positions_lowerbody_10frame_step5",
        positions,
    )
    write_span(
        obs, layout, "motion_joint_velocities_lowerbody_10frame_step5",
        velocities,
    )
    write_span(obs, layout, "vr_3point_local_target", vr_positions)
    write_span(
        obs, layout, "vr_3point_local_orn_target", vr_orientations)
    write_span(obs, layout, "motion_anchor_orientation", anchor)
    return obs[None, :]


def build_smpl_encoder_obs(
    layout, clip, frame, play, base_quat, heading_state
):
    obs = build_empty_encoder_obs(layout, "smpl")
    smpl = gather_motion_values(
        clip.smpl_joints.reshape(clip.num_frames, -1),
        clip, frame, play, 10, 1,
    )
    orientations = compute_anchor_orientations(
        clip, frame, play, 10, 1, base_quat, heading_state)
    wrists = gather_motion_values(
        clip.joint_pos[:, WRIST_POLICY], clip, frame, play, 10, 1)
    write_span(obs, layout, "smpl_joints_10frame_step1", smpl)
    write_span(
        obs, layout, "smpl_anchor_orientation_10frame_step1",
        orientations,
    )
    write_span(
        obs, layout, "motion_joint_positions_wrists_10frame_step1",
        wrists,
    )
    return obs[None, :]


def write_span(obs, layout, name, values):
    span = find_encoder_span(layout, name)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != span.dim:
        raise ValueError(
            f"Observation {name} needs {span.dim} values, got "
            f"{values.size}")
    obs[span.start:span.end] = values


def gather_motion_values(values, clip, frame, play, num_frames, step_size):
    frames = compute_future_frames(
        clip, frame, play, num_frames, step_size)
    return values[frames].reshape(-1).astype(np.float32)


def compute_future_frames(clip, frame, play, num_frames, step_size):
    if not play:
        return np.full((num_frames,), clamp_frame(frame, clip), np.int32)
    frames = frame + np.arange(num_frames, dtype=np.int32) * step_size
    return np.minimum(frames, clip.num_frames - 1)


def compute_anchor_orientations(
    clip, frame, play, num_frames, step_size, base_quat, heading_state
):
    values = []
    apply_delta = compute_apply_delta_heading(heading_state)
    frames = compute_future_frames(
        clip, frame, play, num_frames, step_size)
    for frame_id in frames:
        ref_root = clip.body_quat[frame_id, 0]
        new_ref = quat_mul(apply_delta, ref_root)
        relative = quat_mul(quat_conjugate(base_quat), new_ref)
        values.append(quat_to_sixd(relative))
    return np.concatenate(values).astype(np.float32)


def compute_vr_targets(clip, frame):
    storage_ids = []
    for body_id in VR_BODY_IDS:
        matches = np.flatnonzero(clip.body_indices == body_id)
        if matches.size == 0:
            raise ValueError(f"Motion {clip.name} has no body ID {body_id}")
        storage_ids.append(int(matches[0]))
    root_pos = clip.body_pos[frame, 0]
    root_quat = clip.body_quat[frame, 0]
    root_inv = quat_conjugate(root_quat)
    positions = []
    orientations = []
    for storage_id, offset in zip(storage_ids, VR_OFFSETS):
        body_pos = clip.body_pos[frame, storage_id]
        body_quat = clip.body_quat[frame, storage_id]
        point = body_pos + quat_rotate(body_quat, offset)
        positions.append(quat_rotate(root_inv, point - root_pos))
        orientations.append(quat_mul(root_inv, body_quat))
    args = np.concatenate(positions), np.concatenate(orientations)
    return tuple(value.astype(np.float32) for value in args)


def build_history_buffer():
    return deque(maxlen=HISTORY_FRAMES)


def build_history_entry(qpos, qvel, last_action):
    joint_q = qpos[ROOT_QPOS_OFFSET:ROOT_QPOS_OFFSET + NUM_JOINTS]
    joint_dq = qvel[ROOT_QVEL_OFFSET:ROOT_QVEL_OFFSET + NUM_JOINTS]
    return build_history_entry_from_state(
        qpos[3:7], qvel[3:6], joint_q, joint_dq, last_action)


def build_history_entry_from_state(
    base_quat, base_ang_vel, joint_q, joint_dq, last_action
):
    body_q = joint_q[POLICY_FROM_SCENE]
    body_dq = joint_dq[POLICY_FROM_SCENE]
    body_q = body_q - DEFAULT_ANGLES[POLICY_FROM_SCENE]
    args = (
        np.asarray(base_quat, dtype=np.float32),
        np.asarray(base_ang_vel, dtype=np.float32),
        body_q.astype(np.float32),
        body_dq.astype(np.float32),
        np.asarray(last_action, dtype=np.float32),
    )
    return HistoryEntry(*args)


def build_policy_tail(layout, history):
    entries = list(history)[-HISTORY_FRAMES:]
    while len(entries) < HISTORY_FRAMES:
        entries.insert(0, compute_zero_entry())
    tail_dim = compute_policy_tail_dim(layout)
    policy_tail = np.zeros((tail_dim,), dtype=np.float32)
    for span in layout.policy_spans[1:]:
        values = compute_policy_values(entries, span.name)
        start = span.start - layout.token_dim
        policy_tail[start:start + span.dim] = values
    return policy_tail[None, :]


def compute_policy_values(entries, name):
    if name == "his_base_angular_velocity_10frame_step1":
        return flatten_entry_values(entries, "base_ang_vel")
    if name == "his_body_joint_positions_10frame_step1":
        return flatten_entry_values(entries, "body_q")
    if name == "his_body_joint_velocities_10frame_step1":
        return flatten_entry_values(entries, "body_dq")
    if name == "his_last_actions_10frame_step1":
        return flatten_entry_values(entries, "last_action")
    if name == "his_gravity_dir_10frame_step1":
        values = [compute_gravity_dir(entry.base_quat) for entry in entries]
        return np.concatenate(values).astype(np.float32)
    raise ValueError(f"Unsupported policy observation: {name}")


def flatten_entry_values(entries, field):
    return np.concatenate(
        [getattr(entry, field) for entry in entries]).astype(np.float32)


def compute_zero_entry():
    args = (
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        np.zeros((NUM_JOINTS,), dtype=np.float32),
        np.zeros((NUM_JOINTS,), dtype=np.float32),
        np.zeros((NUM_JOINTS,), dtype=np.float32),
    )
    return HistoryEntry(*args)


def compute_heading_state(base_quat, ref_root_quat, delta_heading=0.0):
    args = (
        np.asarray(base_quat, dtype=np.float32),
        np.asarray(ref_root_quat, dtype=np.float32),
        np.float32(delta_heading),
    )
    return HeadingState(*args)


def compute_apply_delta_heading(heading_state):
    init_heading = compute_heading_quat(heading_state.init_base_quat)
    ref_heading_inv = compute_heading_quat_inv(
        heading_state.init_ref_root_quat)
    apply_delta = quat_mul(init_heading, ref_heading_inv)
    if heading_state.delta_heading != 0.0:
        apply_delta = quat_mul(
            yaw_to_quat(heading_state.delta_heading), apply_delta)
    return apply_delta


def compute_action_targets(action_policy):
    scene_action = np.asarray(action_policy)[SCENE_FROM_POLICY]
    return DEFAULT_ANGLES + scene_action * ACTION_SCALE


def compute_pd_torque(q_target, joint_q, joint_dq):
    torques = KPS * (q_target - joint_q) - KDS * joint_dq
    return np.clip(torques, -EFFORT_LIMITS, EFFORT_LIMITS)


def compute_hand_pd_torque(joint_q, joint_dq):
    torques = -1.5 * joint_q - 0.1 * joint_dq
    return np.clip(torques, -HAND_EFFORT_LIMITS, HAND_EFFORT_LIMITS)


def find_joint_addresses(model):
    body_names = (
        "hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")
    body_joint_ids = []
    hand_joint_ids = []
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name or ""
        if any(part in name for part in body_names):
            body_joint_ids.append(joint_id)
        elif "hand" in name:
            hand_joint_ids.append(joint_id)
    if len(body_joint_ids) != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} SONIC body joints, got "
            f"{len(body_joint_ids)}")
    actuator_by_joint = {
        int(model.actuator_trnid[index, 0]): index
        for index in range(model.nu)
    }

    def addresses(joint_ids):
        joint_ids = np.asarray(joint_ids, dtype=np.int32)
        qpos = model.jnt_qposadr[joint_ids].astype(np.int32)
        dof = model.jnt_dofadr[joint_ids].astype(np.int32)
        actuator = np.asarray(
            [actuator_by_joint[int(index)] for index in joint_ids],
            dtype=np.int32,
        )
        return qpos, dof, actuator

    body = addresses(body_joint_ids)
    hand = addresses(hand_joint_ids)
    return JointAddresses(*(body + hand))


def apply_support_force(model, data, body_id, strength=1.0):
    if strength <= 0.0:
        data.xfrc_applied[body_id] = 0.0
        return
    pose = np.zeros((13,), dtype=np.float64)
    pose[:3] = data.xpos[body_id]
    pose[3:7] = data.xquat[body_id]
    args = model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, pose[7:13], 0
    mujoco.mj_objectVelocity(*args)
    pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
    data.xfrc_applied[body_id] = compute_support_wrench(pose, strength)


def compute_support_wrench(pose, strength=1.0):
    pos = pose[:3]
    quat = pose[3:7]
    lin_vel = pose[7:10]
    ang_vel = pose[10:13]
    force = SUPPORT_KP_POS * (SUPPORT_POINT - pos)
    force = force - SUPPORT_KD_POS * lin_vel
    torque = -SUPPORT_KP_ANG * quat_to_rotvec(quat)
    torque = torque - SUPPORT_KD_ANG * ang_vel
    return np.float64(strength) * np.concatenate([force, torque])


def compute_reference_markers(mode_name, clip, frame, root_pos, root_quat):
    if mode_name == "teleop":
        local_points, _ = compute_vr_targets(clip, frame)
        local_points = local_points.reshape(-1, 3)
    elif mode_name == "smpl":
        local_points = clip.smpl_joints[frame]
    else:
        ref_root_pos = clip.body_pos[frame, 0]
        ref_root_inv = quat_conjugate(clip.body_quat[frame, 0])
        local_points = [
            quat_rotate(ref_root_inv, point - ref_root_pos)
            for point in clip.body_pos[frame]
        ]
    return np.asarray(
        [root_pos + quat_rotate(root_quat, point) for point in local_points])


def update_viewer_markers(viewer, points, mode_name):
    if viewer is None or viewer.user_scn is None:
        return
    points = np.asarray(points)
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = min(points.shape[0], scene.maxgeom)
        colors = compute_marker_colors(mode_name, scene.ngeom)
        radius = 0.045 if mode_name == "teleop" else 0.025
        size = np.asarray([radius, radius, radius], dtype=np.float64)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        for index in range(scene.ngeom):
            mujoco.mjv_initGeom(
                scene.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
                size, points[index], identity, colors[index])


def clear_viewer_markers(viewer):
    if viewer is None or viewer.user_scn is None:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0


def compute_marker_colors(mode_name, count):
    if mode_name == "teleop":
        palette = np.asarray(
            [[0.2, 0.5, 1.0, 0.85], [1.0, 0.3, 0.2, 0.85],
             [1.0, 0.85, 0.1, 0.85]],
            dtype=np.float32,
        )
        return palette[np.arange(count) % len(palette)]
    color = [0.9, 0.2, 1.0, 0.65]
    if mode_name == "g1":
        color = [0.1, 0.85, 1.0, 0.55]
    return np.tile(np.asarray(color, np.float32), (count, 1))


def clamp_frame(frame, clip):
    return max(0, min(int(frame), clip.num_frames - 1))


def compute_gravity_dir(base_quat):
    down = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate(quat_conjugate(base_quat), down)


def quat_to_rotvec(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:]
    sin_half = np.linalg.norm(vector)
    if sin_half < 1e-8:
        return np.zeros((3,), dtype=np.float64)
    axis = vector / sin_half
    angle = 2.0 * np.arctan2(sin_half, quat[0])
    return axis * angle


def compute_heading_quat(quat):
    return yaw_to_quat(compute_yaw(quat))


def compute_heading_quat_inv(quat):
    return quat_conjugate(compute_heading_quat(quat))


def compute_yaw(quat):
    w, x, y, z = quat
    top = 2.0 * (w * z + x * y)
    bottom = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(top, bottom)


def yaw_to_quat(yaw):
    half_yaw = 0.5 * yaw
    return np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )


def quat_conjugate(quat):
    w, x, y, z = quat
    return np.asarray([w, -x, -y, -z], dtype=np.float32)


def quat_mul(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    args = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    return np.asarray(args, dtype=np.float32)


def quat_rotate(quat, vector):
    quat = np.asarray(quat, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    quat_vector = quat[1:]
    first = vector * (2.0 * quat[0] * quat[0] - 1.0)
    second = np.cross(quat_vector, vector) * quat[0] * 2.0
    third = quat_vector * np.dot(quat_vector, vector) * 2.0
    return (first + second + third).astype(np.float32)


def quat_to_sixd(quat):
    matrix = quat_to_matrix(quat)
    return matrix[:, :2].reshape(-1).astype(np.float32)


def quat_to_matrix(quat):
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
