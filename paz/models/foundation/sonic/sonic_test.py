"""Opt-in numeric parity check against the real released SONIC weights.

Skipped unless SONIC_RELEASE_DIR points at a gear_sonic_deploy release
directory containing observation_config.yaml, model_encoder.onnx and
model_decoder.onnx (see the SONIC Keras port handoff for that layout).
"""
import os
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from paz.models.foundation.sonic.conversion import port_sonic_weights
from paz.models.foundation.sonic.layout import compute_mode_scalar_index
from paz.models.foundation.sonic.layout import load_release_observation_layout

RELEASE_DIR = os.environ.get("SONIC_RELEASE_DIR")


def release_files_exist():
    if not RELEASE_DIR:
        return False
    release_dir = Path(RELEASE_DIR)
    names = ("observation_config.yaml", "model_encoder.onnx",
             "model_decoder.onnx")
    return all((release_dir / name).exists() for name in names)


pytestmark = pytest.mark.skipif(
    not release_files_exist(),
    reason="SONIC_RELEASE_DIR not set to a real release directory")


def build_release_runtime():
    release_dir = Path(RELEASE_DIR)
    obs_config = release_dir / "observation_config.yaml"
    encoder_onnx = release_dir / "model_encoder.onnx"
    decoder_onnx = release_dir / "model_decoder.onnx"
    layout = load_release_observation_layout(obs_config)
    encoder, decoder, actor = port_sonic_weights(
        layout, encoder_onnx, decoder_onnx)
    encoder_session = ort.InferenceSession(str(encoder_onnx))
    decoder_session = ort.InferenceSession(str(decoder_onnx))
    return layout, encoder, decoder, encoder_session, decoder_session


def test_encoder_matches_onnx_runtime_per_mode():
    layout, encoder, _, encoder_session, _ = build_release_runtime()
    mode_index = compute_mode_scalar_index(layout)
    rng = np.random.default_rng(0)
    for mode in layout.encoder_modes:
        x = rng.normal(size=(1, layout.encoder_spans[-1].end))
        x = x.astype("float32")
        x[0, mode_index] = float(mode.mode_id)
        onnx_out = encoder_session.run(None, {"obs_dict": x})[0]
        paz_out = np.array(encoder(x, training=False))
        assert np.abs(onnx_out - paz_out).max() < 1e-4


def test_decoder_matches_onnx_runtime():
    layout, _, decoder, _, decoder_session = build_release_runtime()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, layout.policy_spans[-1].end)).astype("float32")
    onnx_out = decoder_session.run(None, {"obs_dict": x})[0]
    paz_out = np.array(decoder(x, training=False))
    assert np.abs(onnx_out - paz_out).max() < 1e-4
