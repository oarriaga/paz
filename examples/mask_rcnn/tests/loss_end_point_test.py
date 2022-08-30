import pytest
import numpy as np
import tensorflow as tf

from mask_rcnn.config import Config
from mask_rcnn.loss_end_point import ProposalBBoxLoss,ProposalClassLoss
from mask_rcnn.loss_end_point import BBoxLoss, ClassLoss, MaskLoss, smooth_L1_loss
from mask_rcnn.loss_end_point import reshape_data, batch_pack_graph
tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()


class MaskRCNNConfig(Config):
    NAME = 'shapes'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3


@pytest.fixture
def config():
    config = MaskRCNNConfig()
    return config


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
                           [180, 179, 211, 205]]], dtype=tf.float32)
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
    y_pred = tf.constant([[[[129, 122, 167, 172],
                            [213, 203, 279, 258]],
                           [[194, 49, 218, 75],
                            [180, 179, 211, 205]],
                           [[217, 137, 295, 220],
                            [214, 110, 243, 159]],
                           [[194, 49, 218, 75],
                            [213, 203, 279, 258]]]], dtype=tf.float32)
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


@pytest.mark.parametrize('softmax_loss', [1.1095])
def test_class_loss(config, class_ids, softmax_loss):
    y_true = tf.ones((1, 4))
    active_class = tf.constant([1., 0., 1.])
    class_loss = ClassLoss(config=config, active_class_ids=active_class)\
        (y_true, class_ids)
    if session._closed:
        class_loss = class_loss.numpy()
    else:
        class_loss = session.run(class_loss)
    assert np.round(class_loss, 4) == np.float32(softmax_loss)


@pytest.mark.parametrize('categorical_loss', [0.6931])
def test_rpn_classifier_loss(config, categorical_loss):
    y_true = tf.ones((1, 4, 1))
    y_pred = tf.zeros((1, 4, 2))
    classifier_loss = ProposalClassLoss(config=config)\
        (y_true, y_pred)
    if session._closed:
        classifier_loss = classifier_loss.numpy()
    else:
        classifier_loss = session.run(classifier_loss)
    assert np.round(classifier_loss, 4) == np.float32(categorical_loss)


@pytest.mark.parametrize('l1_loss', [52.125])
def test_rpn_box_loss(config, target_RPN_boxes, RPN_boxes, l1_loss):
    rpn_match = tf.ones((1, 4, 1))
    box_loss = ProposalBBoxLoss(config=config,rpn_match=rpn_match)\
        (target_RPN_boxes,RPN_boxes)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert box_loss == l1_loss


@pytest.mark.parametrize('l1_loss', [88.75])
def test_mrcnn_box_loss(config, target_boxes, boxes, l1_loss):
    target_ids = tf.constant([1, 0])
    box_loss = BBoxLoss(config=config,target_class_ids=target_ids)\
        (target_boxes, boxes)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert box_loss == l1_loss


@pytest.mark.parametrize('crossentropy_loss', [11.522856])
def test_mrcnn_mask_loss(config, target_mask, mask, crossentropy_loss):
    y_true = target_mask
    target_ids = tf.constant([1, 0])
    mask_loss = MaskLoss(config=config, target_class_ids=target_ids)\
        (y_true, mask)
    if session._closed:
        mask_loss = mask_loss.numpy()
    else:
        mask_loss = session.run(mask_loss)
    assert mask_loss == np.float32(crossentropy_loss)


@pytest.mark.parametrize('l1_loss', [[[19.5, 46.5, 57.5, 24.5],
                                      [32.5,23.5,67.5,52.5]]])
def test_smooth_L1_loss(target_boxes, l1_loss):
    y_pred = tf.constant([[[129, 122, 167, 172],[213, 203, 279, 258]]],
                         dtype=tf.float32)
    box_loss = smooth_L1_loss(target_boxes,y_pred)
    if session._closed:
        box_loss = box_loss.numpy()
    else:
        box_loss = session.run(box_loss)
    assert np.all(box_loss == np.array(l1_loss))


@pytest.mark.parametrize('pred_data', [[[[0., 0.],[1., 0.]],
                                        [[1., 1.],[0., 1.]],
                                        [[0., 1.],[1., 0.]]]])
def test_smooth_reshape(mask, pred_data):
    target_ids = tf.constant([1, 0])
    y_true = tf.constant([[[[0., 1.],[1., 0.],[1., 1.]]]], dtype=tf.float32)
    reshape_data_val = reshape_data(target_ids, y_true, mask)
    if session._closed:
        reshape_data_val = reshape_data_val.numpy()
    else:
        reshape_data_val = session.run(reshape_data_val)
    assert np.all(reshape_data_val[2] == np.array(pred_data))


@pytest.mark.parametrize('pred_data', [[[[129, 122, 167, 172],
                            [213, 203, 279, 258]]]])
def test_batch_graph(boxes,pred_data):
    counts = tf.constant([1])
    num_rows = 1
    values = batch_pack_graph(boxes, counts, num_rows)
    if session._closed:
        values = values.numpy()
    else:
        values = session.run(values)
    assert np.all(values == pred_data)