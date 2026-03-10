import json

import pytest

from examples.speech_to_text.tokenizer import build_token_id_to_bytes
from examples.speech_to_text.tokenizer import decode_token_ids
from examples.speech_to_text.tokenizer import decode_whisper_token_ids
from examples.speech_to_text.tokenizer import find_special_token_id


def test_special_token_lookup_matches_local_assets():
    token_ids = (
        find_special_token_id("<|startoftranscript|>"),
        find_special_token_id("<|endoftext|>"),
        find_special_token_id("<|notimestamps|>"),
        find_special_token_id("<|transcribe|>"),
        find_special_token_id("<|translate|>"),
    )
    assert token_ids == (50257, 50256, 50362, 50357, 50358)


def test_special_tokens_decode_deterministically():
    decoded = decode_whisper_token_ids([50257, 50357, 50362])
    assert decoded == "<|startoftranscript|><|transcribe|><|notimestamps|>"


def test_unknown_token_id_raises_key_error():
    with pytest.raises(KeyError, match="Unknown token id: 999999"):
        decode_whisper_token_ids([999999])


def test_vocabulary_decoding_works_for_controlled_case(tmp_path):
    vocabulary_path = tmp_path / "vocabulary.json"
    with open(vocabulary_path, "w", encoding="utf-8") as filedata:
        json.dump({"hello": 0, ".": 1}, filedata)
    token_id_to_bytes = build_token_id_to_bytes(vocabulary_path)
    decoded = decode_token_ids([0, 1], token_id_to_bytes, {})
    assert decoded == "hello."


def test_real_vocabulary_decoding_is_stable():
    decoded = decode_whisper_token_ids([31373, 13])
    assert decoded == "hello."
