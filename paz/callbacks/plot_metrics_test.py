import os

import numpy as np
import keras

import paz


def build_regression_data():
    inputs = np.random.RandomState(0).randn(8, 3).astype("float32")
    return inputs, inputs.sum(axis=1, keepdims=True)


def test_writes_a_figure_every_epoch(tmp_path):
    x, y = build_regression_data()
    inputs = keras.Input((3,))
    model = keras.Model(inputs, keras.layers.Dense(1)(inputs))
    model.compile("adam", "mse")
    path = os.path.join(str(tmp_path), "metrics.png")
    callback = paz.callbacks.PlotMetrics(path)
    kwargs = dict(epochs=2, verbose=0, validation_data=(x, y))
    model.fit(x, y, callbacks=[callback], **kwargs)
    assert os.path.exists(path)
    assert len(callback.history["loss"]) == 2
    assert len(callback.history["val_loss"]) == 2
