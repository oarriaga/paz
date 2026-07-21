import json

import pytest

from paz.models.foundation.florence2 import tokenizer


@pytest.fixture
def toy_tokenizer(tmp_path):
    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "l": 3, "o": 4,
                  "w": 5, "lo": 6, "low": 7, "Ġ": 8, "Ġlow": 9}
    merges = ["l o", "lo w", "Ġ low"]
    source = {"model": {"vocab": vocabulary, "merges": merges}}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(source))
    return tokenizer.load_tokenizer(path)


def test_encode_wraps_with_bos_and_eos(toy_tokenizer):
    token_ids = tokenizer.encode(toy_tokenizer, "low")
    assert token_ids[0] == 0
    assert token_ids[-1] == 2


def test_encode_applies_merges(toy_tokenizer):
    assert tokenizer.encode(toy_tokenizer, "low") == [0, 7, 2]
    assert tokenizer.encode(toy_tokenizer, "low low") == [0, 7, 9, 2]


def test_encode_truncates_to_max_length(toy_tokenizer):
    token_ids = tokenizer.encode(toy_tokenizer, "low low low", max_length=3)
    assert token_ids == [0, 7, 2]


def test_split_pattern_keeps_underscores():
    pieces = tokenizer.SPLIT_PATTERN.findall("pick_up KITCHEN_SCENE_1")
    assert "".join(pieces) == "pick_up KITCHEN_SCENE_1"


def test_policy_prompt_matches_reference_format():
    prompt = tokenizer.build_policy_prompt("pick up the milk")
    reference = ("Agent Type: 1-arm Franka Panda, Action Space: "
                 "Delta End-Effector,  Task Instruction: pick up the milk")
    assert prompt == reference
