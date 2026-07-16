import os

os.environ["KERAS_BACKEND"] = "jax"

import pytest

from paz.models.foundation.sonic.layout import EncoderModeLayout
from paz.models.foundation.sonic.layout import ObservationSpan
from paz.models.foundation.sonic.layout import SonicObservationLayout


@pytest.fixture
def toy_layout():
    # token_dim must stay 64 (2 * 32) to satisfy compute_release_fsq's
    # fixed reshape in paz.models.foundation.sonic.model.
    token_dim = 64
    mode_span = ObservationSpan("encoder_mode_4", 0, 4, 4)
    flat_span = ObservationSpan("motion_root_z_position", 4, 5, 1)
    temporal_span = ObservationSpan(
        "motion_anchor_orientation_10frame_step5", 5, 65, 60)
    flat_mode = EncoderModeLayout(
        "flat", 0, ("encoder_mode_4", "motion_root_z_position"),
        (flat_span,))
    temporal_mode = EncoderModeLayout(
        "temporal", 1,
        ("encoder_mode_4", "motion_anchor_orientation_10frame_step5"),
        (temporal_span,), temporal_frames=10)
    policy_spans = (
        ObservationSpan("token_state", 0, token_dim, token_dim),
        ObservationSpan("policy_tail", token_dim, token_dim + 6, 6),
    )
    encoder_spans = (mode_span, flat_span, temporal_span)
    modes = (flat_mode, temporal_mode)
    return SonicObservationLayout(policy_spans, encoder_spans, modes,
                                   token_dim)
