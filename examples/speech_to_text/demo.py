import os

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import argparse

import sys
from pathlib import Path

import numpy as np
from keras import ops

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text.audio import load, resample, to_float, to_mono
from examples.speech_to_text.decoding import (
    KVDecoder,
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import (
    decode_token_ids_with_kv_cache,
    extract_text_token_ids,
)
from examples.speech_to_text.model2 import (
    CONFIGS,
    WhisperCrossCache,
    WhisperDecoderStep,
    WhisperEncoder,
    WhisperFrontend,
)
from examples.speech_to_text.tokenizer import decode_whisper_token_ids
from examples.speech_to_text.tokenizer import find_special_token_id


def preprocess_waveform(waveform, sample_rate, expected_sample_rate=16000):
    waveform = to_float(waveform)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, expected_sample_rate)
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = ops.convert_to_tensor(waveform, dtype="float32")
    return expected_sample_rate, waveform


def transcribe(
    wav_path,
    frontend=None,
    encoder=None,
    cross_cache_model=None,
    decoder_step_model=None,
    max_tokens=64,
):
    if frontend is None:
        frontend = WhisperFrontend()
    if encoder is None:
        encoder = WhisperEncoder(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_encoder",
        )
    if cross_cache_model is None:
        cross_cache_model = WhisperCrossCache(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_cross_cache",
        )
    if decoder_step_model is None:
        decoder_step_model = WhisperDecoderStep(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_decoder_step",
        )
    sample_rate, waveform = preprocess(wav_path)
    token_ids, generated_token_ids, decoded_text = transcribe_waveform(
        waveform,
        frontend=frontend,
        encoder=encoder,
        cross_cache_model=cross_cache_model,
        decoder_step_model=decoder_step_model,
        max_tokens=max_tokens,
    )
    return sample_rate, token_ids, generated_token_ids, decoded_text


def transcribe_waveform(
    waveform,
    frontend=None,
    encoder=None,
    cross_cache_model=None,
    decoder_step_model=None,
    decoder=None,
    max_tokens=64,
):
    if frontend is None:
        frontend = WhisperFrontend()
    if encoder is None:
        encoder = WhisperEncoder(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_encoder",
        )
    if cross_cache_model is None:
        cross_cache_model = WhisperCrossCache(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_cross_cache",
        )
    if decoder_step_model is None:
        decoder_step_model = WhisperDecoderStep(
            **CONFIGS["whisper_base_en"],
            weights="whisper_base_en",
            name="whisper_base_en_decoder_step",
        )
    if len(waveform.shape) == 1:
        waveform = ops.expand_dims(waveform, axis=0)
    encoder_features = frontend(waveform)
    encoder_output = encoder(encoder_features)
    prompt_token_ids = build_whisper_base_en_prompt_token_ids()
    stop_token_id = find_special_token_id("<|endoftext|>")
    cache_shape = decoder_step_model.input_shape[1]
    if decoder is None:
        decoder = KVDecoder(decoder_step_model, prompt_token_ids, max_tokens)

    token_ids = decode_token_ids_with_kv_cache(
        decoder,
        cache_shape,
        cross_cache_model,
        encoder_output,
        stop_token_id,
    )
    generated_token_ids = extract_text_token_ids(
        token_ids, len(prompt_token_ids), stop_token_id
    )
    decoded_text = decode_whisper_token_ids(generated_token_ids)
    return token_ids, generated_token_ids, decoded_text


def preprocess(wav_path, expected_sample_rate=16000):
    waveform, sample_rate = load(wav_path)
    return preprocess_waveform(waveform, sample_rate, expected_sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper speech-to-text demo")
    parser.add_argument(
        "--audio_path", default=str(Path(__file__).with_name("harvard.wav"))
    )
    parser.add_argument(
        "--model_name", default="whisper_base_en", choices=["whisper_base_en"]
    )
    parser.add_argument("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    frontend = WhisperFrontend()
    encoder = WhisperEncoder(
        **CONFIGS["whisper_base_en"],
        weights="whisper_base_en",
        name="whisper_base_en_encoder",
    )
    cross_cache_model = WhisperCrossCache(
        **CONFIGS["whisper_base_en"],
        weights="whisper_base_en",
        name="whisper_base_en_cross_cache",
    )
    decoder_step_model = WhisperDecoderStep(
        **CONFIGS["whisper_base_en"],
        weights="whisper_base_en",
        name="whisper_base_en_decoder_step",
    )
    _, _, _, decoded_text = transcribe(
        Path(args.audio_path),
        frontend,
        encoder,
        cross_cache_model,
        decoder_step_model,
        args.max_tokens,
    )
    print(decoded_text)
