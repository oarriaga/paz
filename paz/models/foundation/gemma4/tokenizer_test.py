import json

from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer

# Tiny hand-built byte-BPE tokenizer whose ids are referenced by the asserts
# below. Written to a temp file per test so no JSON asset is committed (the
# repo tracks only source files; real tokenizer.json assets ship with weights).
TEST_TOKENIZER = {
    "normalizer": {"type": "Replace"},
    "decoder": {"type": "Metaspace"},
    "added_tokens": [
        {"content": "<pad>", "special": True},
        {"content": "<eos>", "special": True},
        {"content": "<bos>", "special": True},
        {"content": "<unk>", "special": True},
        {"content": "<mask>", "special": True},
        {"content": "<|turn>", "special": True},
        {"content": "<turn|>", "special": True},
    ],
    "model": {
        "type": "BPE",
        "merges": [],
        "vocab": {
            "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4,
            "<|turn>": 5, "<turn|>": 6, "h": 7, "i": 8, "u": 9, "s": 10,
            "e": 11, "r": 12, "\n": 13, "m": 14, "o": 15, "d": 16, "l": 17,
            "▁": 18, "t": 19, "<0xC3>": 20, "<0xA9>": 21,
        },
    },
}


def build_tokenizer(tmp_path, **kwargs):
    path = tmp_path / "gemma4_test_tokenizer.json"
    path.write_text(json.dumps(TEST_TOKENIZER), encoding="utf-8")
    return Gemma4Tokenizer(path, **kwargs)


def test_json_tokenizer_round_trips_text(tmp_path):
    tokenizer = build_tokenizer(tmp_path)
    text = "hi there"
    token_ids = tokenizer.tokenize(text)
    assert token_ids == [7, 8, 18, 19, 7, 11, 12, 11]
    assert tokenizer.detokenize(token_ids) == text


def test_json_tokenizer_formats_generation_prompt(tmp_path):
    tokenizer = build_tokenizer(tmp_path)
    text = tokenizer.format_generation_prompt("hi")
    assert text == "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
    token_ids = tokenizer.tokenize_generation_prompt("hi")
    assert token_ids == [2, 5, 9, 10, 11, 12, 13, 7, 8, 6, 13, 5,
                         14, 15, 16, 11, 17, 13]
    assert tokenizer.get_stop_token_ids() == (1, 6)


def test_json_tokenizer_supports_batches(tmp_path):
    tokenizer = build_tokenizer(tmp_path)
    texts = ["hi there", "hi"]
    token_ids = tokenizer.tokenize(texts)
    assert token_ids == [[7, 8, 18, 19, 7, 11, 12, 11], [7, 8]]
    assert tokenizer.detokenize(token_ids) == texts


def test_json_tokenizer_supports_byte_tokens(tmp_path):
    tokenizer = build_tokenizer(tmp_path)
    token_ids = tokenizer.tokenize("é")
    assert token_ids == [20, 21]
    assert tokenizer.detokenize(token_ids) == "é"


def test_json_tokenizer_exposes_special_ids(tmp_path):
    tokenizer = build_tokenizer(tmp_path)
    assert tokenizer.start_token_id == tokenizer.token_to_id("<bos>")
    assert tokenizer.end_token_id == tokenizer.token_to_id("<eos>")
    assert tokenizer.pad_token_id == tokenizer.token_to_id("<pad>")
    assert tokenizer.start_of_turn_token_id == tokenizer.token_to_id("<|turn>")
    assert tokenizer.end_of_turn_token_id == tokenizer.token_to_id("<turn|>")
    assert tokenizer.start_of_image_token_id == -1


def test_tokenizer_rejects_non_json_assets(tmp_path):
    path = tmp_path / "tokenizer.spm"
    path.write_text("stub", encoding="utf-8")
    try:
        Gemma4Tokenizer(path)
    except ValueError as error:
        assert "tokenizer.json only" in str(error)
        return
    raise AssertionError("Gemma4Tokenizer should reject non-json assets")
