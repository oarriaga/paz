from pathlib import Path

import numpy as np
import jax
import pytest
from keras import ops

import paz
from paz.models.foundation.whisper import model
from paz.models.foundation.whisper.model import Whisper
from paz.models.foundation.whisper.model import WhisperFrontend
from paz.models.foundation.whisper.model import resolve_weights_path
from paz.models.foundation.whisper.decoding import (
    KVDecoder, kv_decode, build_prompt_token_ids,
    extract_text_token_ids)
from paz.models.foundation.whisper.tokenizer import find_special_token_id
from paz.models.foundation.whisper.tokenizer import decode_tokens
from paz.applications.transcribers import build_transcript_printer

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = ROOT / "examples" / "speech_to_text" / "whisper_models"
AUDIO_PATH = ROOT / "examples" / "speech_to_text" / "harvard.wav"

HARVARD_TOKEN_IDS = [
    383, 39985, 8508, 286, 1468, 6099, 18459, 364, 13, 632, 2753, 4894, 284,
    2222, 503, 262, 28192, 13, 317, 4692, 19550, 45815, 1535, 290, 1976, 395,
    13, 317, 8268, 2298, 293, 18221, 3734, 351, 8891, 13, 26075, 418, 435,
    22175, 389, 616, 4004, 13, 317, 1976, 395, 913]


def has_local_weights_and_audio():
    weights = WEIGHTS_DIR / "whisper_tiny_en" / "encoder.weights.h5"
    return weights.exists() and AUDIO_PATH.exists()


def preprocess(waveform, sample_rate):
    waveform = paz.audio.to_float(waveform)
    waveform = paz.audio.to_mono(waveform)
    waveform = paz.audio.resample(waveform, sample_rate, 16000)
    waveform = np.clip(waveform, -1.0, 1.0)
    return ops.convert_to_tensor(waveform, dtype="float32")


def test_whisper_returns_named_models():
    models = Whisper("whisper_tiny_en", weights=None)
    assert models._fields == (
        "frontend", "encoder", "cross_cache", "decoder_step")
    assert models.encoder.__class__.__name__ == "Functional"
    inputs = [tensor.name for tensor in models.decoder_step.inputs]
    assert inputs[0] == "decoder_token_ids"
    assert inputs[1] == "self_attention_cache"


def test_frontend_feature_shape():
    frontend = WhisperFrontend()
    waveform = ops.zeros((1, 16000), dtype="float32")
    features = frontend(waveform)
    assert features.shape[-1] == 80


def test_unhosted_variant_raises_before_download(monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("download attempted")

    monkeypatch.setattr(model, "get_file", fail_download)
    with pytest.raises(ValueError, match="whisper_tiny_multi"):
        resolve_weights_path("whisper_tiny_multi", "encoder", None)


def test_hosted_variant_reaches_download(monkeypatch):
    requested = []

    def record_download(asset, *args, **kwargs):
        requested.append(asset)
        raise ConnectionError("download stopped by test")

    monkeypatch.setattr(model, "get_file", record_download)
    with pytest.raises(ConnectionError):
        resolve_weights_path("whisper_tiny_en", "encoder", None)
    assert requested == ["whisper_tiny_en_encoder.weights.h5"]


def test_unhosted_variant_resolves_local_path(tmp_path):
    path = resolve_weights_path("whisper_tiny_multi", "encoder", tmp_path)
    assert path == tmp_path / "whisper_tiny_multi" / "encoder.weights.h5"


@pytest.mark.weights
def test_prompt_and_stop_tokens_resolve_from_assets():
    assert build_prompt_token_ids() == [50257, 50357, 50362]
    assert find_special_token_id("<|endoftext|>") == 50256


@pytest.mark.weights
def test_transcript_printer_streams_full_text(capsys):
    stop_id = find_special_token_id("<|endoftext|>")
    print_token = build_transcript_printer(stop_id)
    for token_id in HARVARD_TOKEN_IDS + [stop_id]:
        print_token(token_id)
    streamed = capsys.readouterr().out
    assert streamed == decode_tokens(HARVARD_TOKEN_IDS)


LOCAL_REASON = "local whisper_tiny_en weights or harvard.wav not present"


@pytest.mark.weights
@pytest.mark.skipif(not has_local_weights_and_audio(), reason=LOCAL_REASON)
def test_harvard_token_ids_match_snapshot():
    frontend, encoder, cross_model, decoder_step = Whisper(
        "whisper_tiny_en", weights="pretrained", models_path=str(WEIGHTS_DIR))
    waveform, sample_rate = paz.audio.load(AUDIO_PATH)
    waveform = ops.expand_dims(preprocess(waveform, sample_rate), axis=0)
    encoder_output = encoder(frontend(waveform))
    prompt = build_prompt_token_ids()
    stop_id = find_special_token_id("<|endoftext|>")
    cache_shape = decoder_step.input_shape[1]
    decoder = KVDecoder(decoder_step, prompt, 48)
    key = jax.random.PRNGKey(0)
    ids = kv_decode(key, decoder, cache_shape, cross_model, encoder_output,
                    stop_id)
    text_ids = extract_text_token_ids(ids, len(prompt), stop_id)
    assert text_ids == HARVARD_TOKEN_IDS
