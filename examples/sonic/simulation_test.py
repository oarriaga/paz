import numpy as np
import pytest

pytest.importorskip("mujoco")

from simulation import DEFAULT_ANGLES
from simulation import HistoryEntry
from simulation import MotionClip
from simulation import build_encoder_obs
from simulation import build_history_buffer
from simulation import build_history_entry
from simulation import build_history_entry_from_state
from simulation import build_policy_tail
from simulation import check_mode_available
from simulation import compute_heading_state
from simulation import compute_reference_markers
from simulation import compute_support_wrench
from simulation import compute_hand_pd_torque
from simulation import compute_pd_torque
from simulation import compute_vr_targets
from simulation import load_body_indices
from paz.models.foundation.sonic.layout import EncoderModeLayout
from paz.models.foundation.sonic.layout import ObservationSpan
from paz.models.foundation.sonic.layout import SonicObservationLayout
from paz.models.foundation.sonic.layout import find_encoder_span


def build_spans(names_and_dims, start=0):
    spans = []
    offset = start
    for name, dim in names_and_dims:
        spans.append(ObservationSpan(name, offset, offset + dim, dim))
        offset += dim
    return tuple(spans)


def build_release_layout():
    policy_dims = (
        ("token_state", 64),
        ("his_base_angular_velocity_10frame_step1", 30),
        ("his_body_joint_positions_10frame_step1", 290),
        ("his_body_joint_velocities_10frame_step1", 290),
        ("his_last_actions_10frame_step1", 290),
        ("his_gravity_dir_10frame_step1", 30),
    )
    encoder_dims = (
        ("encoder_mode_4", 4),
        ("motion_joint_positions_10frame_step5", 290),
        ("motion_joint_velocities_10frame_step5", 290),
        ("motion_root_z_position_10frame_step5", 10),
        ("motion_root_z_position", 1),
        ("motion_anchor_orientation", 6),
        ("motion_anchor_orientation_10frame_step5", 60),
        ("motion_joint_positions_lowerbody_10frame_step5", 120),
        ("motion_joint_velocities_lowerbody_10frame_step5", 120),
        ("vr_3point_local_target", 9),
        ("vr_3point_local_orn_target", 12),
        ("smpl_joints_10frame_step1", 720),
        ("smpl_anchor_orientation_10frame_step1", 60),
        ("motion_joint_positions_wrists_10frame_step1", 60),
    )
    modes = tuple(
        EncoderModeLayout(name, mode_id, (), ())
        for mode_id, name in enumerate(("g1", "teleop", "smpl"))
    )
    return SonicObservationLayout(
        build_spans(policy_dims), build_spans(encoder_dims), modes, 64)


def build_clip(include_smpl=True):
    num_frames = 60
    frame_values = np.arange(num_frames, dtype=np.float32)[:, None]
    joint_pos = frame_values + np.arange(29, dtype=np.float32)[None, :]
    joint_vel = np.ones_like(joint_pos)
    body_indices = np.asarray([0, 9, 28, 29], dtype=np.int32)
    body_pos = np.zeros((num_frames, 4, 3), dtype=np.float32)
    body_pos[:, 1] = [0.0, 0.0, 0.7]
    body_pos[:, 2] = [0.3, 0.2, 1.0]
    body_pos[:, 3] = [0.3, -0.2, 1.0]
    body_quat = np.zeros((num_frames, 4, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    smpl_joints = None
    if include_smpl:
        joints = np.arange(72, dtype=np.float32).reshape(1, 24, 3)
        smpl_joints = np.tile(joints, (num_frames, 1, 1))
    args = (
        "synthetic", joint_pos, joint_vel, body_pos, body_quat,
        body_indices, smpl_joints, num_frames,
    )
    return MotionClip(*args)


def build_heading(clip):
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return identity, compute_heading_state(identity, clip.body_quat[0, 0])


@pytest.mark.parametrize("mode_name, mode_id", [
    ("g1", 0), ("teleop", 1), ("smpl", 2),
])
def test_build_encoder_obs_covers_release_modes(mode_name, mode_id):
    layout = build_release_layout()
    clip = build_clip()
    base_quat, heading = build_heading(clip)
    obs = build_encoder_obs(
        layout, mode_name, clip, 2, True, base_quat, heading)
    mode_span = find_encoder_span(layout, "encoder_mode_4")
    assert obs.shape == (1, 1762)
    assert obs[0, mode_span.start] == mode_id
    assert np.isfinite(obs).all()


def test_teleop_targets_match_original_root_local_construction():
    clip = build_clip()
    positions, orientations = compute_vr_targets(clip, 0)
    expected_positions = np.asarray(
        [[0.48, 0.175, 1.0], [0.48, -0.175, 1.0], [0.0, 0.0, 1.05]],
        dtype=np.float32,
    )
    expected_orientations = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1))
    assert np.allclose(positions.reshape(3, 3), expected_positions)
    assert np.array_equal(orientations.reshape(3, 4), expected_orientations)


def test_smpl_mode_is_data_gated():
    clip = build_clip(include_smpl=False)
    available, reason = check_mode_available(clip, "smpl")
    assert available is False
    assert "smpl_joints.csv" in reason


def test_policy_tail_has_release_shape_and_latest_action():
    layout = build_release_layout()
    history = build_history_buffer()
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for index in range(10):
        action = np.full((29,), index, dtype=np.float32)
        entry = HistoryEntry(
            identity, np.zeros(3, np.float32), DEFAULT_ANGLES,
            np.zeros(29, np.float32), action)
        history.append(entry)
    tail = build_policy_tail(layout, history)
    assert tail.shape == (1, 930)
    last_action_span = layout.policy_spans[4]
    start = last_action_span.start - layout.token_dim
    actions = tail[0, start:start + last_action_span.dim].reshape(10, 29)
    assert np.array_equal(actions[-1], np.full(29, 9, np.float32))


def test_reference_marker_counts_match_mode_targets():
    clip = build_clip()
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_pos = np.zeros(3, dtype=np.float32)
    assert compute_reference_markers(
        "g1", clip, 0, root_pos, identity).shape == (4, 3)
    assert compute_reference_markers(
        "teleop", clip, 0, root_pos, identity).shape == (3, 3)
    assert compute_reference_markers(
        "smpl", clip, 0, root_pos, identity).shape == (24, 3)


def test_support_wrench_is_zero_at_assisted_pose():
    pose = np.zeros(13, dtype=np.float64)
    pose[:3] = [0.0, 0.0, 1.0]
    pose[3] = 1.0
    assert np.allclose(compute_support_wrench(pose), 0.0)


def test_load_body_indices_supports_wrapped_metadata(tmp_path):
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "Body part indexes:\n[ 0  4 10 18\n  5 11 19  9 28 29]\n",
        encoding="utf-8",
    )
    expected = np.asarray([0, 4, 10, 18, 5, 11, 19, 9, 28, 29])
    assert np.array_equal(load_body_indices(metadata), expected)


def test_explicit_state_history_matches_contiguous_scene():
    qpos = np.arange(36, dtype=np.float32)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qvel = np.arange(35, dtype=np.float32)
    last_action = np.arange(29, dtype=np.float32)
    contiguous = build_history_entry(qpos, qvel, last_action)
    explicit = build_history_entry_from_state(
        qpos[3:7], qvel[3:6], qpos[7:36], qvel[6:35], last_action)
    for field in HistoryEntry._fields:
        assert np.array_equal(getattr(explicit, field),
                              getattr(contiguous, field))


def test_pd_torque_uses_original_simulator_limits():
    target = np.full(29, 1000.0, dtype=np.float32)
    torque = compute_pd_torque(
        target, np.zeros(29, np.float32), np.zeros(29, np.float32))
    expected = np.asarray(
        [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50,
         88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25,
         25, 5, 5], dtype=np.float32)
    assert np.array_equal(torque, expected)
    hand_torque = compute_hand_pd_torque(
        np.full(14, -1000.0), np.zeros(14))
    expected_hand = np.asarray(
        [2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7] * 2, dtype=np.float32)
    assert np.array_equal(hand_torque, expected_hand)
