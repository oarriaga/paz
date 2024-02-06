import pytest
import numpy as np
from paz.models.pose_estimation import (EfficientPosePhi0, EfficientPosePhi1,
                                        EfficientPosePhi2, EfficientPosePhi3,
                                        EfficientPosePhi4, EfficientPosePhi5,
                                        EfficientPosePhi6, EfficientPosePhi7)
from paz.datasets.linemod import LINEMOD_CAMERA_MATRIX as camera_matrix
from processors import RegressTranslation, ComputeTxTyTz


@pytest.fixture()
def get_camera_matrix():
    return camera_matrix


@pytest.mark.parametrize(
        ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
                    EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
                    EfficientPosePhi6, EfficientPosePhi7])
def test_RegressTranslation(model):
    model = model(num_classes=2, base_weights='COCO', head_weights=None)
    regress_translation = RegressTranslation(model.translation_priors)
    translation_raw = np.zeros_like(model.translation_priors)
    translation_raw = np.expand_dims(translation_raw, axis=0)
    translation = regress_translation(translation_raw)
    assert translation[:, 0].sum() == model.translation_priors[:, 0].sum()
    assert translation[:, 1].sum() == model.translation_priors[:, 1].sum()
    assert translation[:, 2].sum() == translation_raw[:, :, 2].sum()
    del model


@pytest.mark.parametrize(
        ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
                    EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
                    EfficientPosePhi6, EfficientPosePhi7])
def test_ComputeTxTyTz(model, get_camera_matrix):
    model = model(num_classes=2, base_weights='COCO', head_weights=None)
    translation_raw = np.zeros_like(model.translation_priors)
    compute_tx_ty_tz = ComputeTxTyTz()
    tx_ty_tz = compute_tx_ty_tz(translation_raw, get_camera_matrix, 0.8)
    assert tx_ty_tz.shape == model.translation_priors.shape
    del model
