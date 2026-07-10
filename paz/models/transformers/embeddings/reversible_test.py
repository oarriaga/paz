import numpy as np
from keras import ops

from paz.models.transformers.embeddings.reversible import ReversibleEmbedding


def test_forward_looks_up_embeddings():
    layer = ReversibleEmbedding(10, 4)
    layer.build(None)
    tokens = ops.convert_to_tensor([[1, 2, 3]])
    hidden = layer(tokens)
    assert tuple(hidden.shape) == (1, 3, 4)


def test_reverse_projects_back_to_vocabulary():
    layer = ReversibleEmbedding(10, 4)
    layer.build(None)
    hidden = ops.ones((2, 5, 4))
    logits = layer(hidden, reverse=True)
    assert tuple(logits.shape) == (2, 5, 10)


def test_tied_reverse_uses_transposed_embeddings():
    layer = ReversibleEmbedding(10, 4)
    layer.build(None)
    hidden = ops.convert_to_tensor(np.random.default_rng(0).standard_normal((2, 5, 4)), dtype="float32")  # fmt: skip
    logits = np.asarray(layer(hidden, reverse=True))
    expected = np.asarray(hidden) @ np.asarray(layer.embeddings).T
    assert np.allclose(logits, expected, atol=1e-5)
