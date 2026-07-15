import numpy as np
from keras import Input, Model

from paz.models.foundation.dinov2 import embeddings


def build_embedding_model(num_registers):
    images = Input((70, 70, 3), name="pixels")
    tokens = embeddings.build(images, 14, 384, 26, num_registers)
    return Model(images, tokens)


def test_patch_tokens_shape():
    images = Input((70, 70, 3))
    patches = embeddings.build_patch_tokens(images, 14, 384)
    assert tuple(patches.shape) == (None, 25, 384)


def test_embeddings_add_class_token():
    model = build_embedding_model(0)
    output = model(np.zeros((2, 70, 70, 3), "float32"))
    assert tuple(output.shape) == (2, 26, 384)


def test_embeddings_insert_register_tokens():
    model = build_embedding_model(4)
    output = model(np.zeros((2, 70, 70, 3), "float32"))
    assert tuple(output.shape) == (2, 30, 384)


def test_position_embedding_is_added():
    model = build_embedding_model(0)
    model.get_layer("pos_embed").set_weights([np.ones((26, 384), "float32")])
    output = np.array(model(np.zeros((2, 70, 70, 3), "float32")))
    assert np.allclose(output, 1.0)
