import numpy as np
from keras import Input, Model

from paz.models.transformers.embeddings.absolute import embed_position


def test_embed_position_returns_the_first_rows_for_every_batch_item():
    x = Input((3, 8))
    positions = embed_position(x, 16, True, None, "absolute_test")
    model = Model(x, positions)
    table = model.get_layer("absolute_test").get_weights()[0]
    inputs = np.random.default_rng(0).normal(size=(2, 3, 8))
    output = np.array(model(inputs.astype("float32")))
    expected = np.broadcast_to(table[:3], (2, 3, 8))
    assert np.allclose(output, expected, atol=1e-6)
