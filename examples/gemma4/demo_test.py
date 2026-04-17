from pathlib import Path

from .configuration import save_config
from .demo import build_models as build_demo_models
from .demo_e2b import build_models, build_tokenizer, generate
from .inference import Gemma4DecoderStep, Gemma4PerLayerEmbeddingStep
from .model import build_text_backbone_args
from .tokenizer import Gemma4Tokenizer

TEST_DATA = Path(__file__).resolve().with_name("test_data")


def test_demo_rejects_split_weights(tmp_path):
    config = build_text_backbone_args(hidden_size_per_layer_input=2)
    save_config(config, tmp_path / "config.json")
    try:
        build_demo_models(tmp_path)
    except ValueError as error:
        assert "demo_e2b.py" in str(error)
        return
    raise AssertionError("demo.py should reject split Gemma4 weights")


def test_demo_e2b_loads_runtime_assets(tmp_path):
    source = TEST_DATA / "gemma4_test_tokenizer.json"
    target = tmp_path / "tokenizer.json"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    vocab_size = Gemma4Tokenizer(source).vocabulary_size()
    config = build_text_backbone_args(
        vocabulary_size=vocab_size,
        hidden_size_per_layer_input=2,
    )
    save_config(config, tmp_path / "config.json")
    embedding_model = Gemma4PerLayerEmbeddingStep(config)
    embedding_model.save_weights(str(tmp_path / "embedding_step.weights.h5"))
    step_model = Gemma4DecoderStep(config)
    step_model.save_weights(str(tmp_path / "decoder_step.weights.h5"))
    tokenizer = build_tokenizer(tmp_path)
    models = build_models(tmp_path)
    text = generate("hi", models, tokenizer, 1)
    assert models[2] == config
    prompt_ids = tokenizer.tokenize_generation_prompt("hi")
    assert prompt_ids[0] == tokenizer.start_token_id
    assert isinstance(text, str)
