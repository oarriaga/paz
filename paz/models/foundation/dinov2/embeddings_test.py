import numpy as np
from keras import Input, Model

from paz.models.foundation.dinov2.embeddings import build_dinov2_embeddings
from paz.models.foundation.dinov2.embeddings import build_patch_tokens


def build_embedding_model(num_register_tokens):
    images = Input((70, 70, 3), name="pixels")
    tokens = build_dinov2_embeddings(images, 14, 384, 26, num_register_tokens)
    return Model(images, tokens)


def test_patch_tokens_shape():
    images = Input((70, 70, 3))
    patches = build_patch_tokens(images, 14, 384)
    assert tuple(patches.shape) == (None, 25, 384)


def test_embeddings_add_class_token():
    model = build_embedding_model(0)
    output = model(np.zeros((2, 70, 70, 3), "float32"))
    assert tuple(output.shape) == (2, 26, 384)


def test_embeddings_insert_register_tokens():
    model = build_embedding_model(4)
    output = model(np.zeros((2, 70, 70, 3), "float32"))
    assert tuple(output.shape) == (2, 30, 384)


def test_position_embedding_changes_tokens():
    model = build_embedding_model(0)
    output = np.array(model(np.zeros((2, 70, 70, 3), "float32")))
    assert np.any(output != 0.0)
