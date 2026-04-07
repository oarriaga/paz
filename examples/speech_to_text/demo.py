import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

import numpy as np
from keras import ops

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text.audio import load, resample
from examples.speech_to_text.audio import to_float, to_mono
from examples.speech_to_text.decoding import KVDecoder
from examples.speech_to_text.decoding import build_whisper_prompt_token_ids
from examples.speech_to_text.decoding import kv_decode
from examples.speech_to_text.decoding import extract_text_token_ids
from examples.speech_to_text.configuration import CONFIGS
from examples.speech_to_text.configuration import to_model_args
from examples.speech_to_text.model import WHISPER_MODELS_DIR
from examples.speech_to_text.model import WhisperCrossCache
from examples.speech_to_text.model import WhisperDecoderStep
from examples.speech_to_text.model import WhisperEncoder
from examples.speech_to_text.model import WhisperFrontend
from examples.speech_to_text.tokenizer import decode_whisper_tokens
from examples.speech_to_text.tokenizer import find_special_token_id


def transcribe(wav_path, models, max_tokens=64):
    waveform, sample_rate = load(wav_path)
    waveform = preprocess(waveform, sample_rate)
    return transcribe_waveform(waveform, models, max_tokens)


def preprocess(waveform, sample_rate, target=16000):
    waveform = to_float(waveform)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, target)
    waveform = np.clip(waveform, -1.0, 1.0)
    return ops.convert_to_tensor(waveform, dtype="float32")


def transcribe_waveform(waveform, models, max_tokens=64):
    frontend_model, encoder, cross_model, decoder_step = models
    if len(waveform.shape) == 1:
        waveform = ops.expand_dims(waveform, axis=0)
    features = frontend_model(waveform)
    encoder_output = encoder(features)
    prompt_ids = build_whisper_prompt_token_ids()
    stop_id = find_special_token_id("<|endoftext|>")
    cache_shape = decoder_step.input_shape[1]
    decoder = KVDecoder(decoder_step, prompt_ids, max_tokens)
    args = (decoder, cache_shape, cross_model, encoder_output, stop_id)
    token_ids = kv_decode(*args)
    text_ids = extract_text_token_ids(token_ids, len(prompt_ids), stop_id)
    text = decode_whisper_tokens(text_ids)
    return token_ids, text_ids, text


def build_models(model_name, models_path=WHISPER_MODELS_DIR):
    model_args = to_model_args(model_name, models_path)
    encoder_args, cross_cache_args, decoder_args, kwargs = model_args
    encoder_name = f"{model_name}_encoder"
    cross_name = f"{model_name}_cross_cache"
    decoder_name = f"{model_name}_decoder_step"
    frontend_model = WhisperFrontend()
    encoder = WhisperEncoder(*encoder_args, name=encoder_name, **kwargs)
    cross_cache = WhisperCrossCache(*cross_cache_args, name=cross_name, **kwargs)
    decoder_step = WhisperDecoderStep(*decoder_args, name=decoder_name, **kwargs)
    return frontend_model, encoder, cross_cache, decoder_step


if __name__ == "__main__":
    description = "Whisper speech-to-text demo"
    parser = argparse.ArgumentParser(description=description)
    default_audio = str(Path(__file__).with_name("harvard.wav"))
    model_names = list(CONFIGS.keys())
    add = parser.add_argument
    add("--audio_path", default=default_audio)
    add("--model_name", default="whisper_base_en", choices=model_names)
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    models = build_models(args.model_name)
    _, _, text = transcribe(Path(args.audio_path), models, args.max_tokens)
    print(text)
