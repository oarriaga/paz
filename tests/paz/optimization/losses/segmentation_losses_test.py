import pytest
import numpy as np

from paz.optimization import DiceLoss, JaccardLoss, FocalLoss
from paz.optimization.losses.segmentation import compute_focal_loss
from paz.optimization.losses.segmentation import compute_F_beta_score
from paz.optimization.losses.segmentation import compute_jaccard_score


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


def test_focal_loss_full_overlap(full_mask):
    score = compute_focal_loss(full_mask, full_mask).numpy()
    assert np.allclose(0.0, score)


def test_focal_loss_half_overlap(full_mask, half_mask):
    score = compute_focal_loss(full_mask, half_mask).numpy()
    assert np.allclose(1.4390868, np.mean(score))


def test_focal_loss_no_overlap(full_mask, empty_mask):
    score = compute_focal_loss(full_mask, empty_mask).numpy()
    assert np.allclose(2.8781736, score)


def test_focal_loss_class_full_overlap(full_mask):
    score = FocalLoss()(full_mask, full_mask).numpy()
    assert np.allclose(0.0, score)


def test_focal_loss_class_half_overlap(full_mask, half_mask):
    score = FocalLoss()(full_mask, half_mask).numpy()
    assert np.allclose(1.4390868, np.mean(score))


def test_focal_loss_class_no_overlap(full_mask, empty_mask):
    score = FocalLoss()(full_mask, empty_mask).numpy()
    assert np.allclose(2.8781736, score)
