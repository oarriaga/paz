import jax
import numpy as np
from keras import ops

import paz
from paz.models import Whisper
from paz.models.foundation.whisper.decoding import (
    KVDecoder, kv_decode, build_whisper_prompt_token_ids,
    extract_text_token_ids)
from paz.models.foundation.whisper.tokenizer import (
    find_special_token_id, decode_whisper_tokens)


def TranscribeWhisper(model_name="whisper_base_en", max_tokens=64, seed=0,
                      weights="paz", models_path=None, select=None):
    models = Whisper(model_name, weights=weights, models_path=models_path)
    prompt = build_whisper_prompt_token_ids()
    stop_id = find_special_token_id("<|endoftext|>")
    cache_shape = models.decoder_step.input_shape[1]
    decoder = KVDecoder(models.decoder_step, prompt, max_tokens, select=select)
    key = jax.random.PRNGKey(seed)

    def preprocess(waveform, sample_rate):
        waveform = paz.audio.to_float(waveform)
        waveform = paz.audio.to_mono(waveform)
        waveform = paz.audio.resample(waveform, sample_rate, 16000)
        waveform = np.clip(waveform, -1.0, 1.0)
        return ops.convert_to_tensor(waveform, dtype="float32")

    def transcribe(waveform, sample_rate=16000):
        waveform = ops.expand_dims(preprocess(waveform, sample_rate), axis=0)
        encoder_output = models.encoder(models.frontend(waveform))
        args = (key, decoder, cache_shape, models.cross_cache,
                encoder_output, stop_id)
        token_ids = kv_decode(*args)
        text_ids = extract_text_token_ids(token_ids, len(prompt), stop_id)
        return decode_whisper_tokens(text_ids)

    return transcribe


def TranscribeWhisperTinyEN(**kwargs):
    return TranscribeWhisper("whisper_tiny_en", **kwargs)


def TranscribeWhisperBaseEN(**kwargs):
    return TranscribeWhisper("whisper_base_en", **kwargs)


def TranscribeWhisperSmallEN(**kwargs):
    return TranscribeWhisper("whisper_small_en", **kwargs)


def TranscribeWhisperMediumEN(**kwargs):
    return TranscribeWhisper("whisper_medium_en", **kwargs)
