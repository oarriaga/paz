import pytest
from mask_rcnn.config import Config
from mask_rcnn.model import MaskRCNN
from mask_rcnn.layers import ProposalLayer, DetectionTargetLayer
from mask_rcnn.utils import norm_boxes_graph, fpn_classifier_graph
from mask_rcnn.utils import parse_image_meta_graph, build_fpn_mask_graph

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda

from mask_rcnn import loss


class MaskRCNNConfig(Config):
    NAME = 'ycb'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80 + 1


@pytest.fixture
def config():
    return MaskRCNNConfig()


@pytest.fixture
def model(config):
    base_model = MaskRCNN(config=config,
                          model_dir='../../mask_rcnn')
    return base_model


@pytest.fixture
def feature_maps(model):
    return model.keras_model.output


@pytest.fixture
def anchors():
    return Input(shape=[None, 4])


@pytest.fixture
def RPN_model(model, feature_maps):
    return model.RPN(feature_maps)


@pytest.fixture
def proposal_layer(RPN_model, anchors, config):
    _, RPN_class, RPN_box = RPN_model
    return ProposalLayer(proposal_count=2000,
                         nms_threshold=0.7,
                         name='ROI',
                         config=config)([RPN_class, RPN_box, anchors])


@pytest.fixture
def ground_truth_boxes():
    input_image = Input(shape=[None, None, 3])
    input_boxes = Input(shape=[None, 4], dtype=tf.float32)
    boxes = Lambda(lambda x:
                   norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_boxes)
    return boxes


@pytest.fixture
def ground_truth(ground_truth_boxes):
    class_ids = Input(shape=[None], dtype=tf.int32)
    masks = Input(shape=[1024, 1024, None], dtype=bool)
    return [class_ids, ground_truth_boxes, masks]


@pytest.fixture
def predicted_mask(config, target, feature_maps):
    rois, _, _, _ = target
    return build_fpn_mask_graph(rois, feature_maps[:-1],
                                config,
                                train_bn=config.TRAIN_BN)


@pytest.fixture
def predictions(proposal_layer, feature_maps, config, predicted_mask):
    class_logits, _, boxes = fpn_classifier_graph(proposal_layer,
                                                  feature_maps[:-1],
                                                  'inference',
                                                  config=config,
                                                  train_bn=config.TRAIN_BN)
    return [class_logits, boxes, predicted_mask]


@pytest.fixture
def target(proposal_layer, ground_truth, config):
    class_ids, boxes, masks = ground_truth
    target_layer = DetectionTargetLayer(config)
    return target_layer([proposal_layer, class_ids, boxes, masks])


@pytest.fixture
def active_class_ids(config):
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE],
                             name='input_image_meta')
    return Lambda(
                lambda x: parse_image_meta_graph(x)['active_class_ids']
                )(input_image_meta)


def test_loss_function(config, RPN_model, target, predictions,
                       predicted_mask, active_class_ids):
    RPN_class_logits, _, RPN_boxes = RPN_model
    target = target[1:]
    loss_function = loss.Loss(config, [RPN_class_logits, RPN_boxes],
                              target, predictions, active_class_ids)
    output = loss_function.compute_loss()
    print(output)
