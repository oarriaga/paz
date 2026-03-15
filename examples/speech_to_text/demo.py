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
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import compute_argmax_token_id_from_model
from examples.speech_to_text.decoding import (
    decode_token_ids,
    extract_text_token_ids,
)
from examples.speech_to_text.model import (
    WhisperBaseEn,
    WhisperFrontend,
    WhisperTinyEn,
)
from examples.speech_to_text.tokenizer import decode_whisper_token_ids
from examples.speech_to_text.tokenizer import find_special_token_id


def transcribe(wav_path, frontend=None, model=None, max_tokens=64):
    if frontend is None:
        frontend = WhisperFrontend()
    if model is None:
        model = WhisperBaseEn()
    sample_rate, waveform = preprocess(wav_path)
    token_ids, generated_token_ids, decoded_text = transcribe_waveform(
        waveform, frontend, model, max_tokens
    )
    return sample_rate, token_ids, generated_token_ids, decoded_text


def transcribe_waveform(waveform, frontend=None, model=None, max_tokens=64):
    if frontend is None:
        frontend = WhisperFrontend()
    if model is None:
        model = WhisperBaseEn()
    if len(waveform.shape) == 1:
        waveform = ops.expand_dims(waveform, axis=0)
    encoder_features = frontend(waveform)
    prompt_token_ids = build_whisper_base_en_prompt_token_ids()
    stop_token_id = find_special_token_id("<|endoftext|>")

    def next_token(token_ids):
        return compute_argmax_token_id_from_model(
            model, encoder_features, token_ids
        )

    token_ids = decode_token_ids(
        next_token,
        prompt_token_ids,
        stop_token_id,
        max_generated_tokens=max_tokens,
    )
    generated_token_ids = extract_text_token_ids(
        token_ids, len(prompt_token_ids), stop_token_id
    )
    decoded_text = decode_whisper_token_ids(generated_token_ids)
    return token_ids, generated_token_ids, decoded_text


def preprocess(wav_path, expected_sample_rate=16000):
    waveform, sample_rate = load(wav_path)
    waveform = to_float(waveform)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, expected_sample_rate)
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = ops.convert_to_tensor(waveform, dtype="float32")
    return expected_sample_rate, waveform


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
# model = WhisperBaseEn() if args.model_name == "whisper_base_en" else None
model = WhisperTinyEn()
_, _, _, decoded_text = transcribe(
    Path(args.audio_path), frontend, model, args.max_tokens
)
print(decoded_text)
