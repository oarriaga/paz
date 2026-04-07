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
from examples.speech_to_text.configs import CONFIGS
from examples.speech_to_text.model import WHISPER_MODELS_DIR
from examples.speech_to_text.model import WhisperCrossCache
from examples.speech_to_text.model import WhisperDecoderStep
from examples.speech_to_text.model import WhisperEncoder
from examples.speech_to_text.model import WhisperFrontend
from examples.speech_to_text.tokenizer import decode_whisper_tokens
from examples.speech_to_text.tokenizer import find_special_token_id


def preprocess_waveform(waveform, sample_rate, target=16000):
    waveform = to_float(waveform)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, target)
    waveform = np.clip(waveform, -1.0, 1.0)
    return ops.convert_to_tensor(waveform, dtype="float32")


def preprocess(wav_path, target=16000):
    waveform, sample_rate = load(wav_path)
    return preprocess_waveform(waveform, sample_rate, target)


def build_models(model_name, models_dir=WHISPER_MODELS_DIR):
    config = CONFIGS[model_name]
    layers = config["num_layers"]
    heads = config["num_heads"]
    hidden_dim = config["hidden_dim"]
    ffn_dim = config["ffn_dim"]
    dropout = config["dropout"]
    frontend_model = WhisperFrontend()
    mels = config["num_mels"]
    enc_seq = config["max_encoder_sequence_length"]
    ename = f"{model_name}_encoder"
    enc_args = (mels, layers, heads, hidden_dim, ffn_dim, enc_seq, dropout)
    wargs = {"weights": model_name, "models_dir": models_dir}
    encoder = WhisperEncoder(*enc_args, name=ename, **wargs)
    cname = f"{model_name}_cross_cache"
    cc_args = (layers, heads, hidden_dim)
    cross_cache = WhisperCrossCache(*cc_args, name=cname, **wargs)
    sname = f"{model_name}_decoder_step"
    vocab_size = config["vocabulary_size"]
    dec_seq = config["max_decoder_sequence_length"]
    args = (vocab_size, layers, heads, hidden_dim, ffn_dim, dec_seq, dropout)
    decoder_step = WhisperDecoderStep(*args, name=sname, **wargs)
    return frontend_model, encoder, cross_cache, decoder_step


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


def transcribe(wav_path, models, max_tokens=64):
    waveform = preprocess(wav_path)
    return transcribe_waveform(waveform, models, max_tokens)


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
