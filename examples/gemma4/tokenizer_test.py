from pathlib import Path

from .tokenizer import Gemma4Tokenizer

LOCAL_TEST_DATA = Path(__file__).resolve().with_name("test_data")


def build_test_data_path(name):
    return LOCAL_TEST_DATA / name


def test_json_tokenizer_round_trips_text():
    path = build_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    text = "hi there"
    token_ids = tokenizer.tokenize(text)
    assert token_ids == [7, 8, 18, 19, 7, 11, 12, 11]
    assert tokenizer.detokenize(token_ids) == text


def test_json_tokenizer_formats_generation_prompt():
    path = build_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    text = tokenizer.format_generation_prompt("hi")
    assert text == "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
    token_ids = tokenizer.tokenize_generation_prompt("hi")
    assert token_ids == [2, 5, 9, 10, 11, 12, 13, 7, 8, 6, 13, 5,
                         14, 15, 16, 11, 17, 13]
    assert tokenizer.get_stop_token_ids() == (1, 6)


def test_json_tokenizer_supports_batches():
    path = build_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    texts = ["hi there", "hi"]
    token_ids = tokenizer.tokenize(texts)
    assert token_ids == [[7, 8, 18, 19, 7, 11, 12, 11], [7, 8]]
    assert tokenizer.detokenize(token_ids) == texts


def test_json_tokenizer_supports_byte_tokens():
    path = build_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    token_ids = tokenizer.tokenize("é")
    assert token_ids == [20, 21]
    assert tokenizer.detokenize(token_ids) == "é"


def test_json_tokenizer_exposes_special_ids():
    path = build_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
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
