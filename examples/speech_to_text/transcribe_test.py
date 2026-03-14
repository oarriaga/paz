from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from examples.speech_to_text import transcribe
from examples.speech_to_text.model import build_whisper_base_en_waveform_to_features_model
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_decoder_model,
)
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_encoder_model,
)
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir


def test_preprocess_wav_for_whisper_returns_expected_sample_rate(tmp_path):
    wav_path = tmp_path / "resampled.wav"
    waveform = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    wavfile.write(wav_path, 8000, waveform)
    sample_rate, _ = transcribe.preprocess_wav_for_whisper(wav_path)
    assert sample_rate == 16000


def test_preprocess_wav_for_whisper_returns_float32_mono_waveform(tmp_path):
    wav_path = tmp_path / "resampled.wav"
    waveform = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    wavfile.write(wav_path, 8000, waveform)
    sample_rate, waveform = transcribe.preprocess_wav_for_whisper(wav_path)
    assert (sample_rate, waveform.dtype, len(waveform.shape)) == (
        16000,
        "float32",
        1,
    )


def test_transcribe_whisper_base_en_waveform_runs_end_to_end(
    clear_keras_session,
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    waveform = build_whisper_frontend_waveform()
    frontend_model = build_whisper_base_en_waveform_to_features_model()
    encoder_model = build_preset_loaded_whisper_base_en_encoder_model()
    decoder_model = build_preset_loaded_whisper_base_en_decoder_model()
    token_ids, generated_token_ids, decoded_text = (
        transcribe.transcribe_whisper_base_en_waveform(
            waveform,
            frontend_model,
            encoder_model,
            decoder_model,
            max_generated_tokens=8,
        )
    )
    assert (
        token_ids,
        generated_token_ids,
        decoded_text,
    ) == (
        [50257, 50357, 50362, 685, 9148, 15154, 62, 48877, 9399, 60, 50256],
        [685, 9148, 15154, 62, 48877, 9399, 60],
        " [BLANK_AUDIO]",
    )


def test_transcribe_whisper_base_en_wav_runs_end_to_end(
    clear_keras_session, tmp_path
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    wav_path = tmp_path / "waveform.wav"
    waveform = build_whisper_frontend_waveform()
    waveform = np.asarray(waveform)
    wavfile.write(wav_path, 16000, waveform.astype("float32"))
    frontend_model = build_whisper_base_en_waveform_to_features_model()
    encoder_model = build_preset_loaded_whisper_base_en_encoder_model()
    decoder_model = build_preset_loaded_whisper_base_en_decoder_model()
    sample_rate, token_ids, generated_token_ids, decoded_text = (
        transcribe.transcribe_whisper_base_en_wav(
            wav_path,
            frontend_model,
            encoder_model,
            decoder_model,
            max_generated_tokens=8,
        )
    )
    assert (
        sample_rate,
        token_ids,
        generated_token_ids,
        decoded_text,
    ) == (
        16000,
        [50257, 50357, 50362, 764, 50256],
        [764],
        " .",
    )


def test_transcribe_main_returns_harvard_text(clear_keras_session):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    audio_path = Path(__file__).with_name("harvard.wav")
    assert transcribe.main(audio_path) == (
        " The stale smell of old-beer lingers. It takes heat to bring "
        "out the odor. A cold dip restores health and zest. A salt "
        "pickle tastes fine with ham. Tacos al pastor are my favorite. "
        "A zestful food is the hot cross bun."
    )
