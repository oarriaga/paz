from pathlib import Path

import numpy as np
import pytest
from keras import ops
from scipy.io import wavfile

from examples.speech_to_text import demo
from examples.speech_to_text.decoding import (
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import (
    compute_argmax_token_id,
)
from examples.speech_to_text.decoding import (
    compute_argmax_token_id_from_model,
)
from examples.speech_to_text.decoding import (
    compute_argmax_token_id_from_decoder_model,
)
from examples.speech_to_text.decoding import decode_token_ids_with_jax_loop
from examples.speech_to_text.decoding import decode_token_ids
from examples.speech_to_text.decoding import extract_text_token_ids
from examples.speech_to_text.model import Whisper
from examples.speech_to_text.model import WhisperCrossCache
from examples.speech_to_text.model import WhisperDecoder
from examples.speech_to_text.model import WhisperDecoderStep
from examples.speech_to_text.model import WhisperEncoder
from examples.speech_to_text.tokenizer import find_special_token_id
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import find_variant_config
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir


def test_base_en_decode_prompt_is_explicit():
    assert build_whisper_base_en_prompt_token_ids() == [50257, 50357, 50362]


def test_next_token_id_uses_last_active_position():
    logits = np.zeros((1, 3, 5), dtype="float32")
    logits[0, 1, 4] = 10.0
    logits[0, 2, 1] = 20.0
    next_token_id = compute_argmax_token_id(ops.convert_to_tensor(logits), 2)
    assert next_token_id == 4


def test_greedy_decode_stops_on_stop_token():
    initial_token_ids = [1, 2]
    next_token_ids = [9, 8, 7]

    def next_token_function(token_ids):
        return next_token_ids[len(token_ids) - len(initial_token_ids)]

    token_ids = decode_token_ids(next_token_function, initial_token_ids, 8)
    assert token_ids == [1, 2, 9, 8]


def test_greedy_decode_stops_on_max_generated_tokens():
    initial_token_ids = [1, 2]

    def next_token_function(token_ids):
        return 7

    token_ids = decode_token_ids(
        next_token_function,
        initial_token_ids,
        8,
        max_generated_tokens=3,
    )
    assert token_ids == [1, 2, 7, 7, 7]


def test_build_text_token_ids_strips_prompt_and_stop():
    token_ids = [50257, 50357, 50362, 31373, 13, 50256, 11]
    text_token_ids = extract_text_token_ids(token_ids, 3, 50256)
    assert text_token_ids == [31373, 13]


def test_argmax_token_id_from_model_uses_last_active_position():
    model = Whisper(**find_variant_config("whisper_tiny_en"))
    encoder_features = ops.ones((1, 4, 80), dtype="float32")
    token_ids = [1, 2, 3]
    next_token_id = compute_argmax_token_id_from_model(
        model,
        encoder_features,
        token_ids,
    )
    assert isinstance(next_token_id, int)


def test_compiled_greedy_decode_matches_python_loop():
    config = find_variant_config("whisper_tiny_en")
    encoder_model = WhisperEncoder(**config)
    decoder_model = WhisperDecoder(**config)
    encoder_features = ops.ones((1, 4, 80), dtype="float32")
    encoder_output = encoder_model(encoder_features)
    prompt_token_ids = build_whisper_base_en_prompt_token_ids()
    stop_token_id = find_special_token_id("<|endoftext|>")

    def next_token(token_ids):
        return compute_argmax_token_id_from_decoder_model(
            decoder_model, encoder_output, token_ids
        )

    python_token_ids = decode_token_ids(
        next_token,
        prompt_token_ids,
        stop_token_id,
        max_generated_tokens=6,
    )
    compiled_token_ids = decode_token_ids_with_jax_loop(
        decoder_model,
        encoder_output,
        prompt_token_ids,
        stop_token_id,
        max_generated_tokens=6,
    )
    assert compiled_token_ids == python_token_ids


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
