"""Numerical parity against fixtures captured from the torch LLM2Vec stack.

Run ``reference.py`` once in a torch environment, convert the weights with
``conversion.py``, then::

    LLM2VEC_FIXTURES=~/.keras/paz/models/llm2vec_fixtures.npz \\
    LLM2VEC_WEIGHTS_DIR=~/.keras/paz/models/llm2vec \\
    pytest paz/models/foundation/llm2vec/parity_test.py -s
"""
import os
from pathlib import Path

import jax.numpy as jp
import numpy as np
import pytest

from paz.models.foundation.llm2vec import encoding
from paz.models.foundation.llm2vec.pretrained import LLM2VecLlama3, embed

FIXTURES = os.environ.get("LLM2VEC_FIXTURES")
WEIGHTS_DIR = os.environ.get("LLM2VEC_WEIGHTS_DIR")
pytestmark = pytest.mark.skipif(
    FIXTURES is None or WEIGHTS_DIR is None,
    reason="set LLM2VEC_FIXTURES and LLM2VEC_WEIGHTS_DIR")


@pytest.fixture(scope="module")
def fixtures():
    return dict(np.load(Path(FIXTURES).expanduser()))


@pytest.fixture(scope="module")
def llm2vec():
    return LLM2VecLlama3(Path(WEIGHTS_DIR).expanduser())


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(name, actual, expected):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    similarity = cosine(actual, expected)
    print(f"{name}: max abs error {error:.2e}, cosine {similarity:.6f}")
    return error, similarity


@pytest.mark.parametrize("index", [0, 1, 2])
def test_tokens_match(fixtures, llm2vec, index):
    prompt = str(fixtures["prompts"][index])
    tokens = encoding.build_prompt_tokens(llm2vec.tokenizer, prompt)
    token_ids, num_pooled = tokens
    assert token_ids == fixtures[f"bfloat16_{index}_input_ids"][0].tolist()
    assert num_pooled == int(fixtures[f"bfloat16_{index}_embed_mask"].sum())


@pytest.mark.parametrize("index", [0, 1, 2])
def test_hidden_states_match_bfloat16_reference(fixtures, llm2vec, index):
    token_ids = jp.asarray(fixtures[f"bfloat16_{index}_input_ids"], "int32")
    hidden = encoding.forward(llm2vec.model, llm2vec.variables, token_ids)
    table = llm2vec.model.get_layer("token_embedding").embeddings
    embeddings = np.asarray(table, "float32")[np.asarray(token_ids)]
    error, _ = report("embeddings", embeddings,
                      fixtures[f"bfloat16_{index}_embeddings"])
    assert error == 0.0
    error, similarity = report("final hidden", hidden,
                               fixtures[f"bfloat16_{index}_hidden"])
    assert similarity > 0.999


@pytest.mark.parametrize("index", [0, 1, 2])
def test_pooled_embedding_matches(fixtures, llm2vec, index):
    prompt = str(fixtures["prompts"][index])
    pooled = embed(llm2vec, prompt)
    for label, tolerance in (("bfloat16", 0.9995), ("float32", 0.9995)):
        _, similarity = report(f"{label} pooled", pooled,
                               fixtures[f"{label}_{index}_pooled"])
        assert similarity > tolerance
