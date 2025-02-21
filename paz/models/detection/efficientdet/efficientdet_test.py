import pytest
from keras.models import Model

# Import the functions to be tested
from paz.models.detection.efficientdet.efficientdet import (
    EFFICIENTDETD0,
    EFFICIENTDETD1,
    EFFICIENTDETD2,
    EFFICIENTDETD3,
    EFFICIENTDETD4,
    EFFICIENTDETD5,
    EFFICIENTDETD6,
    EFFICIENTDETD7,
    EFFICIENTDETD7x,
)


# Parameterize the test over all model functions.
@pytest.mark.parametrize(
    "model_fn",
    [
        EFFICIENTDETD0,
        EFFICIENTDETD1,
        EFFICIENTDETD2,
        EFFICIENTDETD3,
        EFFICIENTDETD4,
        EFFICIENTDETD5,
        EFFICIENTDETD6,
        EFFICIENTDETD7,
        EFFICIENTDETD7x,
    ],
)
def test_efficientdet_models(model_fn):
    """
    Test that each EfficientDet model can be created without downloading
    weights (by using base_weights=None, head_weights=None) and that the
    model has its prior boxes computed.
    """
    model = model_fn(num_classes=90, base_weights=None, head_weights=None)
    assert isinstance(model, Model)
    # Check that the model has a prior_boxes attribute that is not None.
    assert hasattr(model, "prior_boxes")
    assert model.prior_boxes is not None


if __name__ == "__main__":
    pytest.main([__file__])
