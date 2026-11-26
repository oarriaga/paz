import codecs

import jax
import numpy as np
from keras import ops

import paz
from paz.models import Whisper
from paz.models.foundation.whisper.decoding import (
    KVDecoder, kv_decode, build_prompt_token_ids,
    extract_text_token_ids)
from paz.models.foundation.whisper.tokenizer import (
    find_special_token_id, decode_tokens, build_byte_maps)


def TranscribeWhisper(model_name="whisper_base_en", max_tokens=64, seed=0,
                      weights="pretrained", models_path=None, select=None, emit=None):
    """Builds a Whisper speech transcriber.

    # Arguments
    - `model_name`: Name of the Whisper model, for example `"whisper_base_en"`.
    - `max_tokens`: `max_tokens` passed to the decoder.
    - `seed`: Seed of the JAX PRNG key used for decoding.
    - `weights`: `weights` passed to `paz.models.Whisper`.
    - `models_path`: `models_path` passed to `paz.models.Whisper`.
    - `select`: `select` passed to the decoder.
    - `emit`: Callable receiving each decoded token id. `None` prints the text
      while it is decoded.

    # Returns
    Callable `transcribe(waveform, sample_rate=16000)`. It converts the
    waveform to mono, resamples it to 16000 Hz and returns the transcribed
    text. With the default `emit` it also prints the text while decoding.
    """
    models = Whisper(model_name, weights=weights, models_path=models_path)
    prompt = build_prompt_token_ids()
    stop_id = find_special_token_id("<|endoftext|>")
    streams = emit is None
    if streams:
        emit = build_transcript_printer(stop_id)
    cache_shape = models.decoder_step.input_shape[1]
    decoder = KVDecoder(models.decoder_step, prompt, max_tokens, select=select,
                        emit=emit)
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
        if streams:
            print()
        text_ids = extract_text_token_ids(token_ids, len(prompt), stop_id)
        return decode_tokens(text_ids)

    return transcribe


def build_transcript_printer(stop_id):
    """Token sink that prints Whisper text as it decodes, byte-stream safe.

    An incremental UTF-8 decoder buffers partial multi-byte characters, so a
    character split across two tokens still prints correctly once complete.
    """
    id_to_bytes, special_to_bytes = build_byte_maps()
    incremental = codecs.getincrementaldecoder("utf-8")("replace")

    def print_token(token_id):
        token_int = int(token_id)
        if token_int == stop_id:
            return
        if token_int in id_to_bytes:
            token_bytes = id_to_bytes[token_int]
        else:
            token_bytes = special_to_bytes[token_int]
        print(incremental.decode(token_bytes), end="", flush=True)

    return print_token


def TranscribeWhisperTinyEN(**kwargs):
    """Builds a Whisper Tiny English speech transcriber.

    # Arguments
    - `kwargs`: Forwarded to `TranscribeWhisper` with `model_name` set to
      `"whisper_tiny_en"`.

    # Returns
    Callable `transcribe(waveform, sample_rate=16000)`. It converts the
    waveform to mono, resamples it to 16000 Hz and returns the transcribed
    text. With the default `emit` it also prints the text while decoding.
    """
    return TranscribeWhisper("whisper_tiny_en", **kwargs)


def TranscribeWhisperBaseEN(**kwargs):
    """Builds a Whisper Base English speech transcriber.

    # Arguments
    - `kwargs`: Forwarded to `TranscribeWhisper` with `model_name` set to
      `"whisper_base_en"`.

    # Returns
    Callable `transcribe(waveform, sample_rate=16000)`. It converts the
    waveform to mono, resamples it to 16000 Hz and returns the transcribed
    text. With the default `emit` it also prints the text while decoding.
    """
    return TranscribeWhisper("whisper_base_en", **kwargs)


def TranscribeWhisperSmallEN(**kwargs):
    """Builds a Whisper Small English speech transcriber.

    # Arguments
    - `kwargs`: Forwarded to `TranscribeWhisper` with `model_name` set to
      `"whisper_small_en"`.

    # Returns
    Callable `transcribe(waveform, sample_rate=16000)`. It converts the
    waveform to mono, resamples it to 16000 Hz and returns the transcribed
    text. With the default `emit` it also prints the text while decoding.
    """
    return TranscribeWhisper("whisper_small_en", **kwargs)


def TranscribeWhisperMediumEN(**kwargs):
    """Builds a Whisper Medium English speech transcriber.

    # Arguments
    - `kwargs`: Forwarded to `TranscribeWhisper` with `model_name` set to
      `"whisper_medium_en"`.

    # Returns
    Callable `transcribe(waveform, sample_rate=16000)`. It converts the
    waveform to mono, resamples it to 16000 Hz and returns the transcribed
    text. With the default `emit` it also prints the text while decoding.
    """
    return TranscribeWhisper("whisper_medium_en", **kwargs)
