from keras import ops

from paz.models.transformers.embeddings.absolute import embed_position


def test_embed_position_broadcasts_to_input_shape():
    x = ops.zeros((1, 3, 8))
    positions = embed_position(x, 16, True, None, "absolute_test")
    assert tuple(positions.shape) == (1, 3, 8)
