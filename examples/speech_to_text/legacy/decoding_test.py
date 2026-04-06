from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from examples.speech_to_text import demo
from examples.speech_to_text.decoding import (
    KVDecoder,
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import extract_text_token_ids
from examples.speech_to_text.model2 import WhisperCrossCache
from examples.speech_to_text.model2 import WhisperDecoderStep
from examples.speech_to_text.model2 import WhisperEncoder
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import find_variant_config
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir


def test_base_en_decode_prompt_is_explicit():
    assert build_whisper_base_en_prompt_token_ids() == [50257, 50357, 50362]


def test_extract_text_token_ids_strips_prompt_and_stop():
    token_ids = [50257, 50357, 50362, 31373, 13, 50256, 11]
    text_token_ids = extract_text_token_ids(token_ids, 3, 50256)
    assert text_token_ids == [31373, 13]


def test_kv_decoder_captures_max_decode_length(clear_keras_session):
    config = find_variant_config("whisper_tiny_en")
    decoder = KVDecoder(WhisperDecoderStep(**config), [1, 2, 3], 4)
    assert decoder.max_decode_length == 7


def test_preprocess_wav_for_whisper_returns_expected_sample_rate(tmp_path):
    wav_path = tmp_path / "resampled.wav"
    waveform = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    wavfile.write(wav_path, 8000, waveform)
    sample_rate, _ = demo.preprocess(wav_path)
    assert sample_rate == 16000


def test_preprocess_wav_for_whisper_returns_float32_mono_waveform(tmp_path):
    wav_path = tmp_path / "resampled.wav"
    waveform = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    wavfile.write(wav_path, 8000, waveform)
    sample_rate, waveform = demo.preprocess(wav_path)
    assert (sample_rate, waveform.dtype, len(waveform.shape)) == (
        16000,
        "float32",
        1,
    )


def test_preprocess_waveform_matches_wav_semantics():
    waveform = np.array([[0, 32767], [-32768, 0]], dtype=np.int16)
    sample_rate, waveform = demo.preprocess_waveform(waveform, 8000)
    assert (sample_rate, waveform.dtype, len(waveform.shape)) == (
        16000,
        "float32",
        1,
    )


def test_transcribe_waveform_runs_end_to_end(
    clear_keras_session,
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    waveform = build_whisper_frontend_waveform()
    config = find_variant_config("whisper_base_en")
    token_ids, generated_token_ids, decoded_text = (
        demo.transcribe_waveform(
            waveform,
            encoder=WhisperEncoder(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_encoder",
            ),
            cross_cache_model=WhisperCrossCache(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_cross_cache",
            ),
            decoder_step_model=WhisperDecoderStep(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_decoder_step",
            ),
            max_tokens=8,
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


def test_transcribe_waveform_reuses_prebuilt_decoder(
    clear_keras_session,
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    waveform = build_whisper_frontend_waveform()
    config = find_variant_config("whisper_base_en")
    decoder_step_model = WhisperDecoderStep(
        **config,
        weights="whisper_base_en",
        name="whisper_base_en_decoder_step",
    )
    decoder = KVDecoder(
        decoder_step_model,
        build_whisper_base_en_prompt_token_ids(),
        8,
    )
    token_ids, generated_token_ids, decoded_text = demo.transcribe_waveform(
        waveform,
        encoder=WhisperEncoder(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_encoder",
        ),
        cross_cache_model=WhisperCrossCache(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_cross_cache",
        ),
        decoder_step_model=decoder_step_model,
        decoder=decoder,
        max_tokens=8,
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


def test_transcribe_wav_runs_end_to_end(
    clear_keras_session, tmp_path
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    wav_path = tmp_path / "waveform.wav"
    waveform = build_whisper_frontend_waveform()
    waveform = np.asarray(waveform)
    wavfile.write(wav_path, 16000, waveform.astype("float32"))
    config = find_variant_config("whisper_base_en")
    sample_rate, token_ids, generated_token_ids, decoded_text = (
        demo.transcribe(
            wav_path,
            encoder=WhisperEncoder(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_encoder",
            ),
            cross_cache_model=WhisperCrossCache(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_cross_cache",
            ),
            decoder_step_model=WhisperDecoderStep(
                **config,
                weights="whisper_base_en",
                name="whisper_base_en_decoder_step",
            ),
            max_tokens=8,
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


def test_transcribe_returns_harvard_text(
    clear_keras_session,
):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    audio_path = Path(__file__).with_name("harvard.wav")
    config = find_variant_config("whisper_base_en")
    _, _, _, decoded_text = demo.transcribe(
        audio_path,
        encoder=WhisperEncoder(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_encoder",
        ),
        cross_cache_model=WhisperCrossCache(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_cross_cache",
        ),
        decoder_step_model=WhisperDecoderStep(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_decoder_step",
        ),
    )
    assert decoded_text == (
        " The stale smell of old-beer lingers. It takes heat to bring "
        "out the odor. A cold dip restores health and zest. A salt "
        "pickle tastes fine with ham. Tacos al pastor are my favorite. "
        "A zestful food is the hot cross bun."
    )
