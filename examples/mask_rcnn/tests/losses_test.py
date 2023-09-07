import pytest
import numpy as np
import tensorflow as tf

from mask_rcnn.losses.proposal_class_loss import ProposalClassLoss
from mask_rcnn.losses.proposal_bounding_box_loss import ProposalBoundingBoxLoss
from mask_rcnn.losses.proposal_bounding_box_loss import batch_pack_graph
from mask_rcnn.losses.proposal_bounding_box_loss import smooth_L1_loss
from mask_rcnn.model.layers.class_loss import ClassLoss
from mask_rcnn.model.layers.bounding_box_loss import BoundingBoxLoss
from mask_rcnn.model.layers.mask_loss import MaskLoss

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()


@pytest.fixture
def class_ids():
    y_pred = tf.constant([[[0.6, 0.8, 0.7],
                           [0.3, 0.2, 0.2],
                           [0.4, 0.9, 0.1],
                           [0.7, 0.5, 0.2]]], dtype=tf.float32)
    return y_pred


@pytest.fixture
def target_RPN_boxes():
    y_true = tf.constant([[[149, 75, 225, 147],
                           [217, 137, 295, 220],
                           [214, 110, 243, 159],
                           [180, 179, 211, 205],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0]]], dtype=tf.float32)
    return y_true


@pytest.fixture
def RPN_boxes():
    y_pred = tf.constant([[[129, 122, 167, 172],
                           [194, 49, 218, 75],
                           [180, 179, 211, 205],
                           [213, 203, 279, 258]]], dtype=tf.float32)
    return y_pred


@pytest.fixture
def target_boxes():
    y_true = tf.constant([[[149, 75, 225, 147],
                           [180, 179, 211, 205]]], dtype=tf.float32)
    return y_true


@pytest.fixture
def boxes():
    y_pred = tf.constant([[[129, 122, 167, 172],
                           [213, 203, 279, 258]],
                          [[194, 49, 218, 75],
                           [180, 179, 211, 205]],
                          [[217, 137, 295, 220],
                           [214, 110, 243, 159]],
                          [[194, 49, 218, 75],
                           [213, 203, 279, 258]]], dtype=tf.float32)
    return y_pred


@pytest.fixture
def target_mask():
    y_true = tf.constant([[[[0., 1.],
                            [1., 0.]]]], dtype=tf.float32)
    return y_true


@pytest.fixture
def mask():
    y_pred = tf.constant([[[[[0., 1., 0.],
                             [0., 1., 1.]],
                            [[1., 0., 1.],
                             [0., 1., 0.]]]]], dtype=tf.float32)
    return y_pred


@pytest.mark.parametrize('softmax_loss', [0.9854])
def test_class_loss(class_ids, softmax_loss):
    y_true = tf.ones((1, 4))
    # active_class = tf.constant([1., 0., 1.])
    class_loss = ClassLoss(4)(y_true, class_ids)
    if session._closed:
        class_loss = class_loss.numpy()
    else:
        class_loss = session.run(class_loss)
    assert np.round(class_loss, 4) == np.float32(softmax_loss)


@pytest.mark.parametrize('categorical_loss', [0.6931])
def test_rpn_classifier_loss(categorical_loss):
    y_true = tf.ones((1, 4, 1))
    y_pred = tf.zeros((1, 4, 2))
    classifier_loss = ProposalClassLoss()(y_true, y_pred)
    if session._closed:
        classifier_loss = classifier_loss.numpy()
    else:
        classifier_loss = session.run(classifier_loss)
    assert np.round(classifier_loss, 4) == np.float32(categorical_loss)


@pytest.mark.parametrize('l1_loss', [52.125])
def test_rpn_box_loss(target_RPN_boxes, RPN_boxes, l1_loss):
    box_loss = ProposalBoundingBoxLoss(4, 1)(target_RPN_boxes, RPN_boxes)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert box_loss == l1_loss


@pytest.mark.parametrize('l1_loss', [66.75])
def test_mrcnn_box_loss(target_boxes, boxes, l1_loss):
    target_ids = tf.ones([1, 2])
    box_loss = BoundingBoxLoss()([target_boxes, target_ids], boxes)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert box_loss == l1_loss


@pytest.mark.parametrize('crossentropy_loss', [11.522856])
def test_mrcnn_mask_loss(target_mask, mask, crossentropy_loss):
    target_ids = tf.constant([1, 0])
    mask_loss = MaskLoss()([target_mask, target_ids], mask)
    if session._closed:
        mask_loss = mask_loss.numpy()
    else:
        mask_loss = session.run(mask_loss)
    assert mask_loss == np.float32(crossentropy_loss)


@pytest.mark.parametrize('l1_loss', [[[19.5, 46.5, 57.5, 24.5],
                                      [32.5, 23.5, 67.5, 52.5]]])
def test_smooth_L1_loss(target_boxes, l1_loss):
    y_pred = tf.constant([[[129, 122, 167, 172], [213, 203, 279, 258]]],
                         dtype=tf.float32)
    box_loss = smooth_L1_loss(target_boxes, y_pred)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert np.all(box_loss == np.array(l1_loss))


@pytest.mark.parametrize('pred_data', [[[129, 122, 167, 172]]])
def test_batch_graph(boxes, pred_data):
    counts = tf.constant([1])
    num_rows = 1
    values = batch_pack_graph(boxes, counts, num_rows)
    if session._closed:
        values = values.numpy()
    else:
        values = session.run(values)
    assert np.all(values == pred_data)
