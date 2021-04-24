import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda

from mask_rcnn.config import Config
from mask_rcnn.layers import ProposalLayer, DetectionTargetLayer
from mask_rcnn.layers import DetectionLayer, PyramidROIAlign
from mask_rcnn.utils import norm_boxes_graph, fpn_classifier_graph
from mask_rcnn.model import MaskRCNN


class MaskRCNNConfig(Config):
    NAME = 'ycb'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80 + 1


@pytest.fixture
def config():
    config = MaskRCNNConfig()
    config.WINDOW = norm_boxes_graph((171, 0, 853, 1024), (640, 640))
    return config


@pytest.fixture
def model():
    base_model = MaskRCNN(config=MaskRCNNConfig(),
                          model_dir='../../mask_rcnn')
    return base_model


@pytest.fixture
def feature_maps(model):
    return model.keras_model.output


@pytest.fixture
def RPN_model(model, feature_maps):
    return model.RPN(feature_maps)


@pytest.fixture
def anchors():
    return Input(shape=[None, 4])


@pytest.fixture
def ground_truth(ground_truth_boxes):
    class_ids = Input(shape=[None], dtype=tf.int32)
    masks = Input(shape=[1024, 1024, None], dtype=bool)
    return [class_ids, ground_truth_boxes, masks]


@pytest.fixture
def ground_truth_boxes():
    input_image = Input(shape=[None, None, 3])
    input_boxes = Input(shape=[None, 4], dtype=tf.float32)
    boxes = Lambda(lambda x:
                   norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_boxes)
    return boxes


@pytest.fixture
def FPN_classifier(proposal_layer, feature_maps, config):
    _, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(proposal_layer,
                                                      feature_maps[:-1],
                                                      config=config,
                                                      train_bn=config.TRAIN_BN)
    return mrcnn_class, mrcnn_bbox


@pytest.fixture
def proposal_layer(RPN_model, anchors, config):
    _, RPN_class, RPN_box = RPN_model
    return ProposalLayer(proposal_count=2000,
                         nms_threshold=0.7,
                         name='ROI',
                         config=config)([RPN_class, RPN_box, anchors])


@pytest.fixture
def detection_target_layer(proposal_layer, ground_truth, config):
    class_ids, boxes, masks = ground_truth
    target_layer = DetectionTargetLayer(config)
    return target_layer([proposal_layer, class_ids, boxes, masks])


@pytest.mark.parametrize('ROI_shape', [(3,)])
def test_proposal_layer(proposal_layer, ROI_shape):
    num_coordinates = 4
    assert proposal_layer.shape[2] == num_coordinates
    assert K.shape(proposal_layer).shape, ROI_shape


@pytest.mark.parametrize('shapes', [[(3,), (2,), (3,), (4,)]])
def test_detection_target_layer(detection_target_layer, shapes):
    ROIs, target_class, target_box, target_mask = detection_target_layer
    mask_shape = (28, 28)
    results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
                     K.shape(target_box).shape, K.shape(target_mask).shape]
    assert shapes == results_shape
    assert target_mask.shape[-2:] == mask_shape
    assert ROIs.shape[2] == target_box.shape[2] == 4


@pytest.mark.parametrize('shape', [(1, 100, 6)])
def test_detection_layer(proposal_layer, FPN_classifier, config, shape):
    mrcnn_class, mrcnn_bbox = FPN_classifier
    detections = DetectionLayer(config, name='mrcnn_detection')(
                        [proposal_layer, mrcnn_class, mrcnn_bbox])
    assert detections.shape == shape


@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align(proposal_layer, feature_maps, shape):
    ROI_align = PyramidROIAlign([7, 7])(
                                [proposal_layer, shape] + feature_maps)
    assert K.int_shape(ROI_align) == (1, None, 7, 7, 256)

