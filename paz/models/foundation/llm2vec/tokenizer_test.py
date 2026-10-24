import json

import pytest

from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer

LLAMA_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"
                 r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
                 r"|\s+(?!\S)|\s+")


def build_source():
    pieces = ["a", "b", "c", "Ġ", "Ċ", ".", "ab", "Ġa", "Ġab", "ĊĊ"]
    vocabulary = {piece: index for index, piece in enumerate(pieces)}
    added = [{"content": "<|begin_of_text|>", "id": 100, "special": True},
             {"content": "<|eot_id|>", "id": 101, "special": True}]
    return {
        "model": {"type": "BPE", "vocab": vocabulary,
                  "merges": ["a b", "Ġ a", "Ġ ab", "Ċ Ċ"]},
        "added_tokens": added,
        "pre_tokenizer": {"pretokenizers": [
            {"type": "Split", "pattern": {"Regex": LLAMA_PATTERN}}]},
    }


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(build_source()))
    return llama_tokenizer.load_tokenizer(path)


def test_merges_follow_rank_order(tokenizer):
    assert llama_tokenizer.encode(tokenizer, "ab ab") == [6, 8]
    assert llama_tokenizer.encode(tokenizer, "ab a") == [6, 7]


def test_special_tokens_are_matched_verbatim(tokenizer):
    ids = llama_tokenizer.encode(tokenizer, "ab<|eot_id|>")
    assert ids == [6, 101]
    ids = llama_tokenizer.encode(tokenizer, "<|begin_of_text|>ab\n\nc.")
    assert ids == [100, 6, 9, 2, 5]


def test_unknown_bytes_raise(tokenizer):
    with pytest.raises(KeyError):
        llama_tokenizer.encode(tokenizer, "z")
