import numpy as np
import pytest
from paz.optimization.losses.multi_box_loss import MultiBoxLoss


@pytest.fixture
def loss():
    return MultiBoxLoss()


@pytest.fixture
def y_true():
    y_true = np.array(
        [[38.38629, 48.666668, 10.362101, 11.512976, 0., 1.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0.],
         [27.143208, 34.41253, 8.629259, 9.7801285, 1., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0.],
         [27.143208, 68.82506, 8.629259, 13.245829, 1., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0.]], dtype='float32')
    return np.expand_dims(y_true, axis=0)


@pytest.fixture
def y_pred():
    y_pred = np.array(
        [[36.99653894, 46.4176432, 10.35266677, 10.1656072, 0.05621409,
            0.98060555, 0.01017545, 0.03181472, 0.02227341, 0.00503445,
            0.00746015, 0.15980312, 0.10174269, 0.01672697, 0.0111077,
            0.02144868, 0.07346129, 0.03899017, 0.01071656, 0.03946776,
            0.0830264, 0.06763985, 0.04077367, 0.07804006, 0.04347721],
         [26.614379, 32.0909085, 4.2000501, 7.0869583, 0.0423508, 0.91125538,
            0.04441671, 0.03053759, 0.07411292, 0.03454058, 0.04849431,
            0.0592223, 0.0134144, 0.09800261, 0.0433236, 0.04486571,
            0.01135817, 0.08123691, 0.02096761, 0.03070671, 0.04680151,
            0.12522466, 0.06783583, 0.05873021, 0.01260151],
         [2.16936564, 4.4787911, 6.314962, 4.42737758, 0.83406942,
            0.04166197, 0.01605819, 0.04750001, 0.01329675, 0.0126452,
            0.02085183, 0.0172693, 0.03088947, 0.02661936, 0.01231482,
            0.04099588, 0.02453831, 0.07038483, 0.06579002, 0.13424149,
            0.04614118, 0.03297557, 0.1374058, 0.15315633, 0.02119431]],
        dtype='float32')
    return np.expand_dims(y_pred, axis=0)


@pytest.fixture
def target_multibox_loss():
    return 6.8489789962768555


@pytest.fixture
def target_smooth_l1_loss():
    return np.array([[3.5220284, 8.989227, 98.507996]], dtype='float32')


@pytest.fixture
def target_cross_entropy_loss():
    return np.array([[0.019584997, 3.161768, 0.18143862]], dtype='float32')


@pytest.fixture
def target_localization_loss():
    return np.array(3.4861877, dtype='float32')


@pytest.fixture
def target_positive_classification_loss():
    return np.array(0.019584997, dtype='float32')


@pytest.fixture
def target_negative_classification_loss():
    return np.array(3.3432066, dtype='float32')


def test_multiboxloss(y_true, y_pred, loss, target_multibox_loss):
    total_loss = loss.compute_loss(y_true, y_pred)
    assert (float(total_loss) == target_multibox_loss)


def test_smooth_l1_loss(y_true, y_pred, loss, target_smooth_l1_loss):
    smooth_l1_loss = loss._smooth_l1(y_true, y_pred)
    smooth_l1_loss = np.asarray(smooth_l1_loss, dtype='float32')
    print(smooth_l1_loss - target_smooth_l1_loss)
    assert np.allclose(smooth_l1_loss, target_smooth_l1_loss)


def test_cross_entropy_loss(y_true, y_pred, loss, target_cross_entropy_loss):
    cross_entropy_loss = loss._cross_entropy(y_true, y_pred)
    cross_entropy_loss = np.asarray(cross_entropy_loss, dtype='float32')
    assert np.allclose(cross_entropy_loss, target_cross_entropy_loss)


def test_localization_loss(y_true, y_pred, loss, target_localization_loss):
    localization_loss = loss.localization(y_true, y_pred)
    localization_loss = np.asarray(localization_loss, dtype='float32')
    assert np.allclose(localization_loss, target_localization_loss)


def test_positive_classification_loss(y_true, y_pred, loss,
                                      target_positive_classification_loss):
    positive_classification_loss = loss.positive_classification(y_true, y_pred)
    positive_classification_loss = np.asarray(
        positive_classification_loss, dtype='float32')
    assert np.allclose(
        positive_classification_loss, target_positive_classification_loss)


def test_negative_classification_loss(y_true, y_pred, loss,
                                      target_negative_classification_loss):
    negative_classification_loss = loss.negative_classification(y_true, y_pred)
    negative_classification_loss = np.asarray(
        negative_classification_loss, dtype='float32')
    assert np.allclose(
        negative_classification_loss, target_negative_classification_loss)
