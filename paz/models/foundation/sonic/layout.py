"""Observation-layout helpers for the ported SONIC deploy actor."""

from collections import namedtuple
from pathlib import Path
import re

import yaml

_MULTI_FRAME_PATTERN = re.compile(
    r"^(?P<base>.+)_(?P<frames>\d+)frame_step\d+$")

_BASE_OBSERVATION_DIMS = {
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    "his_base_angular_velocity": 3,
    "his_body_joint_positions": 29,
    "his_body_joint_velocities": 29,
    "his_gravity_dir": 3,
    "his_last_actions": 29,
    "motion_anchor_orientation": 6,
    "motion_joint_positions": 29,
    "motion_joint_positions_lowerbody": 12,
    "motion_joint_positions_wrists": 6,
    "motion_joint_velocities": 29,
    "motion_joint_velocities_lowerbody": 12,
    "motion_root_z_position": 1,
    "smpl_anchor_orientation": 6,
    "smpl_joints": 72,
    "token_state": 0,
    "vr_3point_local_orn_target": 12,
    "vr_3point_local_target": 9,
}

_RELEASE_ENCODER_GROUPS = {
    "g1": (
        ("motion_joint_positions_10frame_step5",
         "motion_joint_velocities_10frame_step5"),
        ("motion_anchor_orientation_10frame_step5",),
    ),
    "smpl": (
        ("smpl_joints_10frame_step1",),
        ("smpl_anchor_orientation_10frame_step1",),
        ("motion_joint_positions_wrists_10frame_step1",),
    ),
}

ObservationSpan = namedtuple("ObservationSpan", "name start end dim")

EncoderModeLayout = namedtuple(
    "EncoderModeLayout",
    "name mode_id required_observations feature_spans "
    "temporal_frames feature_groups",
    defaults=(None, ()),
)

SonicObservationLayout = namedtuple(
    "SonicObservationLayout",
    "policy_spans encoder_spans encoder_modes token_dim "
    "action_dim mode_observation_name",
    defaults=(29, "encoder_mode_4"),
)


def compute_mode_input_dim(mode):
    return sum(span.dim for span in mode.feature_spans)


def compute_policy_input_dim(layout):
    if not layout.policy_spans:
        return 0
    return layout.policy_spans[-1].end


def compute_encoder_input_dim(layout):
    if not layout.encoder_spans:
        return 0
    return layout.encoder_spans[-1].end


def compute_decoder_input_dim(layout):
    return compute_policy_input_dim(layout)


def compute_policy_tail_dim(layout):
    return compute_policy_input_dim(layout) - layout.token_dim


def compute_mode_scalar_index(layout):
    mode_span = find_encoder_span(layout, layout.mode_observation_name)
    return mode_span.start


def find_policy_span(layout, name):
    return find_span(layout.policy_spans, name)


def find_encoder_span(layout, name):
    return find_span(layout.encoder_spans, name)


def find_span(spans, name):
    for span in spans:
        if span.name == name:
            return span
    raise KeyError(f"Unknown observation: {name}")


def load_release_observation_layout(obs_config_path):
    config = load_yaml(obs_config_path)
    encoder_config = config.get("encoder", {})
    token_dim = int(encoder_config["dimension"])
    policy_spans = load_enabled_spans(config.get("observations", []), token_dim)
    encoder_spans = load_enabled_spans(
        encoder_config.get("encoder_observations", []), 0)
    check_policy_spans(policy_spans, token_dim)
    check_encoder_spans(encoder_spans)
    encoder_span_map = {span.name: span for span in encoder_spans}
    encoder_modes = load_encoder_modes(encoder_config, encoder_span_map)
    return SonicObservationLayout(
        policy_spans, encoder_spans, encoder_modes, token_dim)


def load_encoder_modes(encoder_config, encoder_span_map):
    encoder_modes = []
    for mode_config in encoder_config.get("encoder_modes", []):
        mode = load_encoder_mode(mode_config, encoder_span_map)
        encoder_modes.append(mode)
    if not encoder_modes:
        raise ValueError("No encoder_modes found in observation config")
    sorted_modes = sorted(encoder_modes, key=lambda mode: mode.mode_id)
    return tuple(sorted_modes)


def load_yaml(obs_config_path):
    obs_config_path = Path(obs_config_path)
    with obs_config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_policy_spans(policy_spans, token_dim):
    if not policy_spans:
        raise ValueError("No enabled policy observations found")
    if policy_spans[0].name != "token_state":
        name = policy_spans[0].name
        raise ValueError(f"Expected 'token_state' first, got {name}")
    if policy_spans[0].dim != token_dim:
        dim = policy_spans[0].dim
        raise ValueError(f"token_state dim mismatch: {dim} != {token_dim}")


def check_encoder_spans(encoder_spans):
    if not encoder_spans:
        raise ValueError("No enabled encoder observations found")
    if encoder_spans[0].name != "encoder_mode_4":
        name = encoder_spans[0].name
        raise ValueError(f"Expected 'encoder_mode_4' first, got {name}")


def load_encoder_mode(mode_config, encoder_span_map):
    required_observations = tuple(
        mode_config.get("required_observations", []))
    feature_spans = []
    for name in required_observations:
        if name != "encoder_mode_4":
            feature_spans.append(encoder_span_map[name])
    feature_spans = tuple(feature_spans)
    temporal_frames = compute_temporal_frames(required_observations)
    feature_groups = compute_feature_groups(
        mode_config["name"], feature_spans, encoder_span_map)
    return EncoderModeLayout(
        mode_config["name"], int(mode_config["mode_id"]),
        required_observations, feature_spans, temporal_frames,
        feature_groups)


def compute_temporal_frames(required_observations):
    frame_counts = []
    for name in required_observations:
        if name == "encoder_mode_4":
            continue
        match = _MULTI_FRAME_PATTERN.match(name)
        if match is None:
            return None
        frame_counts.append(int(match.group("frames")))
    if frame_counts and len(set(frame_counts)) == 1:
        return frame_counts[0]
    return None


def compute_feature_groups(mode_name, feature_spans, encoder_span_map):
    if mode_name not in _RELEASE_ENCODER_GROUPS:
        return tuple((span,) for span in feature_spans)
    feature_groups = []
    for group in _RELEASE_ENCODER_GROUPS[mode_name]:
        spans = tuple(encoder_span_map[name] for name in group)
        feature_groups.append(spans)
    return tuple(feature_groups)


def resolve_observation_dim(name, token_dim):
    if name == "token_state":
        return token_dim
    if name in _BASE_OBSERVATION_DIMS:
        return _BASE_OBSERVATION_DIMS[name]
    match = _MULTI_FRAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported observation name: {name}")
    base_name = match.group("base")
    if base_name not in _BASE_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported base observation name: {base_name}")
    return _BASE_OBSERVATION_DIMS[base_name] * int(match.group("frames"))


def load_enabled_spans(observation_entries, token_dim):
    spans, offset = [], 0
    for entry in observation_entries:
        if not entry.get("enabled", False):
            continue
        dim = resolve_observation_dim(entry["name"], token_dim)
        span = ObservationSpan(entry["name"], offset, offset + dim, dim)
        spans.append(span)
        offset += dim
    return tuple(spans)
