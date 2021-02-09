import pytest
import numpy as np

from model import UNET_VGG16, UNET_VGG19, UNET_RESNET50
from loss import compute_jaccard_score, JaccardLoss
from loss import compute_F_beta_score, DiceLoss


def test_shapes_of_UNETVGG19():
    model = UNET_VGG19(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]


def test_shapes_of_UNETVGG16():
    model = UNET_VGG16(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]


def test_shapes_of_UNET_RESNET50V2():
    model = UNET_RESNET50(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]


@pytest.fixture
def empty_mask():
    return np.zeros((32, 128, 128, 1), dtype=np.float32)


@pytest.fixture
def full_mask():
    return np.ones((32, 128, 128, 1), dtype=np.float32)


@pytest.fixture
def half_mask():
    half_full = np.ones((16, 128, 128, 1))
    half_empty = np.zeros((16, 128, 128, 1))
    half_mask = np.concatenate([half_full, half_empty], axis=0)
    return half_mask.astype(np.float32)


def test_jaccard_score_full_overlap(full_mask):
    score = compute_jaccard_score(full_mask, full_mask).numpy()
    assert np.allclose(1.0, score)


def test_jaccard_score_half_overlap(full_mask, half_mask):
    score = compute_jaccard_score(full_mask, half_mask).numpy()
    assert np.allclose(0.5, np.mean(score))


def test_jaccard_score_no_overlap(full_mask, empty_mask):
    score = compute_jaccard_score(full_mask, empty_mask).numpy()
    assert np.allclose(0.0, score)


def test_jaccard_loss_full_overlap(full_mask):
    score = JaccardLoss()(full_mask, full_mask).numpy()
    assert np.allclose(0.0, score)


def test_jaccard_loss_half_overlap(full_mask, half_mask):
    score = JaccardLoss()(full_mask, half_mask).numpy()
    assert np.allclose(0.5, np.mean(score))


def test_jaccard_loss_no_overlap(full_mask, empty_mask):
    score = JaccardLoss()(full_mask, empty_mask).numpy()
    assert np.allclose(1.0, score)


def test_F_beta_score_full_overlap(full_mask):
    score = compute_F_beta_score(full_mask, full_mask).numpy()
    assert np.allclose(1.0, score)


def test_F_beta_score_half_overlap(full_mask, half_mask):
    score = compute_F_beta_score(full_mask, half_mask).numpy()
    assert np.allclose(0.5, np.mean(score))


def test_F_beta_score_no_overlap(full_mask, empty_mask):
    score = compute_F_beta_score(full_mask, empty_mask).numpy()
    assert np.allclose(0.0, score)


def test_F_beta_loss_full_overlap(full_mask):
    score = DiceLoss()(full_mask, full_mask).numpy()
    assert np.allclose(0.0, score)


def test_F_beta_loss_half_overlap(full_mask, half_mask):
    score = DiceLoss()(full_mask, half_mask).numpy()
    assert np.allclose(0.5, np.mean(score))


def test_F_beta_loss_no_overlap(full_mask, empty_mask):
    score = DiceLoss()(full_mask, empty_mask).numpy()
    assert np.allclose(1.0, score)
