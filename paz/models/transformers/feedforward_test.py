import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras import Input, Model

from paz.models.transformers import feedforward


def test_gelu_shape_and_layer_names():
    x = Input((4, 8))
    y = feedforward.gelu(x, 16, 8, "ff_intermediate", "ff_output")
    model = Model(x, y)
    assert model.output_shape == (None, 4, 8)
    names = [layer.name for layer in model.layers]
    assert "ff_intermediate" in names and "ff_output" in names


def test_glu_shape_and_layer_names():
    x = Input((4, 8))
    y = feedforward.glu(x, 16, 8, "ff_gate", "ff_up", "ff_down")
    model = Model(x, y)
    assert model.output_shape == (None, 4, 8)
    names = [layer.name for layer in model.layers]
    assert all(n in names for n in ("ff_gate", "ff_up", "ff_down"))
