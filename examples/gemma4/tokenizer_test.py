from pathlib import Path

from .reference import build_reference_gemma4_tokenizer
from .reference import build_reference_sentencepiece_tokenizer
from .reference import to_python
from .tokenizer import Gemma4Tokenizer

KERAS_HUB_TEST_DATA = (
    Path(__file__).resolve().parents[3]
    / "keras-hub"
    / "keras_hub"
    / "src"
    / "tests"
    / "test_data"
)
LOCAL_TEST_DATA = Path(__file__).resolve().with_name("test_data")


def build_test_data_path(name):
    return KERAS_HUB_TEST_DATA / name


def build_local_test_data_path(name):
    return LOCAL_TEST_DATA / name


def test_gemma4_word_model_tokens_match_reference():
    proto_path = build_test_data_path("gemma4_test_vocab.spm")
    runtime = Gemma4Tokenizer(proto_path)
    reference = build_reference_gemma4_tokenizer(proto_path)
    text = "the quick brown fox"
    runtime_ids = runtime.tokenize(text)
    reference_ids = to_python(reference(text))
    assert runtime_ids == [9, 14, 10, 12]
    assert runtime_ids == reference_ids


def test_gemma4_word_model_detokenize_matches_reference():
    proto_path = build_test_data_path("gemma4_test_vocab.spm")
    runtime = Gemma4Tokenizer(proto_path)
    reference = build_reference_gemma4_tokenizer(proto_path)
    token_ids = [9, 14, 10, 12]
    runtime_text = runtime.detokenize(token_ids)
    reference_text = to_python(reference.detokenize(token_ids))
    assert runtime_text == "the quick brown fox"
    assert runtime_text == reference_text


def test_gemma4_special_token_ids_match_reference():
    proto_path = build_test_data_path("gemma4_test_vocab.spm")
    runtime = Gemma4Tokenizer(proto_path)
    reference = build_reference_gemma4_tokenizer(proto_path)
    runtime_ids = (
        runtime.start_token_id,
        runtime.end_token_id,
        runtime.pad_token_id,
        runtime.start_of_image_token_id,
        runtime.image_placeholder_id,
        runtime.end_of_image_token_id,
    )
    reference_ids = (
        reference.start_token_id,
        reference.end_token_id,
        reference.pad_token_id,
        reference.start_of_image_token_id,
        reference.image_placeholder_id,
        reference.end_of_image_token_id,
    )
    assert runtime_ids == reference_ids


def test_sentencepiece_bpe_tokens_match_reference():
    proto_path = build_test_data_path("gemma_export_vocab.spm")
    runtime = Gemma4Tokenizer(proto_path, has_vision_tokens=False)
    reference = build_reference_sentencepiece_tokenizer(proto_path)
    texts = ["The quick brown fox jumped.", "I like pizza.", "This is a test."]
    for text in texts:
        runtime_ids = runtime.tokenize(text)
        reference_ids = to_python(reference(text))
        assert runtime_ids == reference_ids


def test_sentencepiece_bpe_detokenize_matches_reference():
    proto_path = build_test_data_path("gemma_export_vocab.spm")
    runtime = Gemma4Tokenizer(proto_path, has_vision_tokens=False)
    reference = build_reference_sentencepiece_tokenizer(proto_path)
    texts = ["The quick brown fox jumped.", "I like pizza."]
    for text in texts:
        token_ids = runtime.tokenize(text)
        runtime_text = runtime.detokenize(token_ids)
        reference_text = to_python(reference.detokenize(token_ids))
        assert runtime_text == reference_text


def test_hf_json_tokenizer_round_trips_text():
    path = build_local_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    text = "hi there"
    token_ids = tokenizer.tokenize(text)
    assert token_ids == [7, 8, 18, 19, 7, 11, 12, 11]
    assert tokenizer.detokenize(token_ids) == text


def test_hf_json_tokenizer_formats_generation_prompt():
    path = build_local_test_data_path("gemma4_test_tokenizer.json")
    tokenizer = Gemma4Tokenizer(path)
    text = tokenizer.format_generation_prompt("hi")
    assert text == "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
    token_ids = tokenizer.tokenize_generation_prompt("hi")
    assert token_ids == [2, 5, 9, 10, 11, 12, 13, 7, 8, 6, 13, 5,
                         14, 15, 16, 11, 17, 13]
    assert tokenizer.get_stop_token_ids() == (1, 6)
