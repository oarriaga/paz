import numpy as np
from keras import Input, Model

from paz.models.detection.rf_detr import projector


def build_projector_model(num_maps=4, hidden_size=16, out_channels=8):
    features = [Input((5, 7, hidden_size)) for _ in range(num_maps)]
    output = projector.build(features, out_channels, 2, "projector")
    return Model(features, output)


def make_inputs(num_maps=4, hidden_size=16, batch=2):
    random = np.random.RandomState(0)
    shape = (batch, 5, 7, hidden_size)
    return [random.randn(*shape).astype("float32") for _ in range(num_maps)]


def test_fuses_maps_into_one_feature_map():
    output = build_projector_model()(make_inputs())
    assert tuple(output.shape) == (2, 5, 7, 8)


def test_cross_stage_concatenates_every_intermediate():
    x = Input((5, 7, 16))
    model = Model(x, projector.build_cross_stage(x, 8, 3, "projector"))
    names = {layer.name for layer in model.layers}
    assert {"projector_m_0_cv1_conv", "projector_m_2_cv2_conv"} <= names
    assert "projector_m_3_cv1_conv" not in names


def test_output_is_layer_normalized():
    output = np.array(build_projector_model()(make_inputs()))
    assert np.allclose(np.mean(output, axis=-1), 0.0, atol=1e-4)
    assert np.allclose(np.std(output, axis=-1), 1.0, atol=1e-3)
