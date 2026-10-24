import json

import jax.numpy as jp
import numpy as np

import paz
from paz.models.foundation.llm2vec import encoding
from paz.models.foundation.llm2vec import model as encoder
from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer
from paz.models.foundation.llm2vec.model_test import TINY
from paz.models.foundation.llm2vec.tokenizer_test import build_source


def build_tiny_tokenizer(tmp_path):
    source = build_source()
    for index, content in enumerate(("<|start_header_id|>",
                                     "<|end_header_id|>", "user")):
        source["added_tokens"].append(
            {"content": content, "id": 110 + index, "special": True})
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(source))
    return llama_tokenizer.load_tokenizer(path)


def test_prompt_tokens_follow_the_llm2vec_layout(tmp_path):
    tokenizer = build_tiny_tokenizer(tmp_path)
    token_ids, num_pooled = encoding.build_prompt_tokens(tokenizer, "ab a ")
    assert token_ids == [100, 110, 112, 111, 9, 6, 7, 101]
    assert num_pooled == 3


def test_pool_averages_the_last_tokens_in_bfloat16():
    hidden = jp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.01]], "bfloat16")
    pooled = encoding.pool(hidden, 2)
    assert pooled.dtype == jp.float32
    assert np.allclose(np.asarray(pooled), [4.0, 5.0], atol=1e-2)


def test_embed_matches_direct_model_call(tmp_path):
    tokenizer = build_tiny_tokenizer(tmp_path)
    model = encoder.build(**dict(TINY, vocab_size=128))
    variables = paz.snapshot_variables(model)
    embedding = encoding.embed(model, variables, tokenizer, "ab a")
    token_ids, num_pooled = encoding.build_prompt_tokens(tokenizer, "ab a")
    hidden = np.asarray(model(np.array([token_ids], "int32")))[0]
    expected = hidden[-num_pooled:].mean(axis=0)
    assert embedding.shape == (TINY["hidden_dim"],)
    assert np.allclose(np.asarray(embedding), expected, atol=1e-5)
