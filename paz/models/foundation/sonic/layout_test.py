import pytest

from paz.models.foundation.sonic.layout import compute_decoder_input_dim
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_mode_scalar_index
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.layout import find_encoder_span
from paz.models.foundation.sonic.layout import load_release_observation_layout

_SYNTHETIC_YAML = """
observations:
  - name: "token_state"
    enabled: true
  - name: "his_last_actions_10frame_step1"
    enabled: true
  - name: "his_gravity_dir"
    enabled: false

encoder:
  dimension: 8
  encoder_observations:
    - name: "encoder_mode_4"
      enabled: true
    - name: "motion_root_z_position"
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5"
      enabled: true
  encoder_modes:
    - name: "beta"
      mode_id: 1
      required_observations:
        - encoder_mode_4
        - motion_root_z_position
    - name: "alpha"
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_anchor_orientation_10frame_step5
"""


def write_synthetic_config(tmp_path, text=_SYNTHETIC_YAML):
    config_path = tmp_path / "observation_config.yaml"
    config_path.write_text(text)
    return config_path


def test_layout_dims_match_release_semantics(tmp_path):
    layout = load_release_observation_layout(write_synthetic_config(tmp_path))
    assert layout.token_dim == 8
    assert compute_policy_tail_dim(layout) == 290
    assert compute_decoder_input_dim(layout) == 298
    assert compute_encoder_input_dim(layout) == 65
    assert compute_mode_scalar_index(layout) == 0


def test_encoder_modes_are_sorted_by_mode_id(tmp_path):
    layout = load_release_observation_layout(write_synthetic_config(tmp_path))
    assert [mode.name for mode in layout.encoder_modes] == ["alpha", "beta"]


def test_temporal_frames_resolved_per_mode(tmp_path):
    layout = load_release_observation_layout(write_synthetic_config(tmp_path))
    alpha, beta = layout.encoder_modes
    assert alpha.temporal_frames == 10
    assert beta.temporal_frames is None


def test_find_encoder_span_raises_on_unknown_name(tmp_path):
    layout = load_release_observation_layout(write_synthetic_config(tmp_path))
    with pytest.raises(KeyError):
        find_encoder_span(layout, "not_a_real_observation")


def test_disabled_observations_are_skipped(tmp_path):
    layout = load_release_observation_layout(write_synthetic_config(tmp_path))
    names = [span.name for span in layout.policy_spans]
    assert "his_gravity_dir" not in names


def test_missing_token_state_first_raises(tmp_path):
    text = _SYNTHETIC_YAML.replace(
        '  - name: "token_state"\n    enabled: true\n', "")
    with pytest.raises(ValueError):
        load_release_observation_layout(write_synthetic_config(tmp_path, text))


def test_missing_encoder_mode_4_first_raises(tmp_path):
    text = _SYNTHETIC_YAML.replace(
        '    - name: "encoder_mode_4"\n      enabled: true\n', "")
    with pytest.raises(ValueError):
        load_release_observation_layout(write_synthetic_config(tmp_path, text))


def test_no_encoder_modes_raises(tmp_path):
    text = _SYNTHETIC_YAML.split("  encoder_modes:")[0]
    with pytest.raises(ValueError):
        load_release_observation_layout(write_synthetic_config(tmp_path, text))


def test_release_encoder_groups_override_g1_grouping(tmp_path):
    text = """
observations:
  - name: "token_state"
    enabled: true

encoder:
  dimension: 4
  encoder_observations:
    - name: "encoder_mode_4"
      enabled: true
    - name: "motion_joint_positions_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_10frame_step5"
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5"
      enabled: true
  encoder_modes:
    - name: "g1"
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_10frame_step5
        - motion_joint_velocities_10frame_step5
        - motion_anchor_orientation_10frame_step5
"""
    layout = load_release_observation_layout(
        write_synthetic_config(tmp_path, text))
    g1 = layout.encoder_modes[0]
    assert len(g1.feature_groups) == 2
    assert len(g1.feature_groups[0]) == 2
    assert len(g1.feature_groups[1]) == 1
