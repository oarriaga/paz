import numpy as np
import pytest
from keras import ops
from scipy.io import wavfile

from examples.speech_to_text.decoding import build_text_token_ids
from examples.speech_to_text.decoding import build_whisper_base_en_decode_prompt
from examples.speech_to_text.decoding import compute_greedy_decode
from examples.speech_to_text.decoding import compute_next_token_id
from examples.speech_to_text.decoding import transcribe_whisper_base_en_wav
from examples.speech_to_text.decoding import transcribe_whisper_base_en_waveform
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


def test_base_en_decode_prompt_is_explicit():
    assert build_whisper_base_en_decode_prompt() == [50257, 50357, 50362]


def test_next_token_id_uses_last_active_position():
    logits = np.zeros((1, 3, 5), dtype="float32")
    logits[0, 1, 4] = 10.0
    logits[0, 2, 1] = 20.0
    next_token_id = compute_next_token_id(ops.convert_to_tensor(logits), 2)
    assert next_token_id == 4


def test_greedy_decode_stops_on_stop_token():
    initial_token_ids = [1, 2]
    next_token_ids = [9, 8, 7]

    def next_token_function(token_ids):
        return next_token_ids[len(token_ids) - len(initial_token_ids)]

    token_ids = compute_greedy_decode(next_token_function, initial_token_ids, 8)
    assert token_ids == [1, 2, 9, 8]


def test_greedy_decode_stops_on_max_generated_tokens():
    initial_token_ids = [1, 2]

    def next_token_function(token_ids):
        return 7

    token_ids = compute_greedy_decode(
        next_token_function,
        initial_token_ids,
        8,
        max_generated_tokens=3,
    )
    assert token_ids == [1, 2, 7, 7, 7]


def test_build_text_token_ids_strips_prompt_and_stop():
    token_ids = [50257, 50357, 50362, 31373, 13, 50256, 11]
    text_token_ids = build_text_token_ids(token_ids, 3, 50256)
    assert text_token_ids == [31373, 13]


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
        transcribe_whisper_base_en_waveform(
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


def test_transcribe_whisper_base_en_wav_runs_end_to_end(clear_keras_session, tmp_path):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    wav_path = tmp_path / "waveform.wav"
    waveform = build_whisper_frontend_waveform()
    waveform = ops.convert_to_numpy(waveform)
    wavfile.write(wav_path, 16000, waveform.astype("float32"))
    frontend_model = build_whisper_base_en_waveform_to_features_model()
    encoder_model = build_preset_loaded_whisper_base_en_encoder_model()
    decoder_model = build_preset_loaded_whisper_base_en_decoder_model()
    sample_rate, token_ids, generated_token_ids, decoded_text = (
        transcribe_whisper_base_en_wav(
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
