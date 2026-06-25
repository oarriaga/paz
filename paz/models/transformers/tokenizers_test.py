import pytest

from paz.models.transformers.tokenizers import build_character_to_byte
from paz.models.transformers.tokenizers import decode_token_ids


def test_character_to_byte_covers_every_byte():
    character_to_byte = build_character_to_byte()
    assert sorted(character_to_byte.values()) == list(range(256))


def test_decode_token_ids_prefers_vocabulary_then_special():
    id_to_bytes = {0: b"he", 1: b"llo"}
    special_to_bytes = {99: "<|eot|>".encode("utf-8")}
    assert decode_token_ids([0, 1], id_to_bytes, special_to_bytes) == "hello"
    text = decode_token_ids([0, 99], id_to_bytes, special_to_bytes)
    assert text == "he<|eot|>"


def test_decode_token_ids_raises_on_unknown():
    with pytest.raises(KeyError):
        decode_token_ids([7], {}, {})
