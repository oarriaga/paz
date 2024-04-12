import pytest
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda


from mask_rcnn.model.layers.proposal import ProposalLayer
from mask_rcnn.model.layers.detection_target import DetectionTargetLayer
from mask_rcnn.model.layers.detection import DetectionLayer
from mask_rcnn.model.layers.pyramid_ROI_align import PyramidROIAlign
from mask_rcnn.model.layers.feature_pyramid_network import FPN_classifier_graph

from mask_rcnn.model.model import MaskRCNN
from mask_rcnn.model.RPN_model import RPN_model
from mask_rcnn.backend.boxes import normalized_boxes


@pytest.fixture
def model():
    window = normalized_boxes((171, 0, 853, 1024), (640, 640))
    base_model = MaskRCNN(model_dir='../../mask_rcnn',
                          image_shape=[128, 128, 3], backbone="resnet101",
                          batch_size=8, images_per_gpu=1,
                          RPN_anchor_scales=(32, 64, 128, 256, 512),
                          train_ROIs_per_image=200,
                          num_classes=81, window=window)
    return base_model


@pytest.fixture
def feature_maps(model):
    return model.keras_model.output


@pytest.fixture
def RPN_model_call(feature_maps):
    return RPN_model(1, [0.5, 1, 2], 256, feature_maps)


@pytest.fixture
def anchors():
    return Input(shape=[None, 4])


@pytest.fixture
def ground_truth(ground_truth_boxes):
    class_ids = Input(shape=[None], dtype=tf.float32)
    masks = Input(shape=[1024, 1024, None], dtype=bool)
    return [class_ids, ground_truth_boxes, masks]


@pytest.fixture
def ground_truth_boxes():
    input_image = Input(shape=[None, None, 3])
    input_boxes = Input(shape=[None, 4], dtype=tf.float32)
    boxes = Lambda(lambda x:
                   normalized_boxes(x, K.shape(input_image)[1:3]))(input_boxes)
    return boxes


@pytest.fixture
def FPN_classifier(proposal_layer, feature_maps):
    _, mrcnn_class, mrcnn_bbox = FPN_classifier_graph(proposal_layer,
                                                      feature_maps[:-1], 81,
                                                      [128, 128, 3])
    return mrcnn_class, mrcnn_bbox


@pytest.fixture
def proposal_layer(RPN_model_call, anchors):
    _, RPN_class, RPN_box = RPN_model_call
    return ProposalLayer(proposal_count=2000, nms_threshold=0.7, name='ROI',
                         RPN_bounding_box_std_dev=np.array([0.1, 0.1,
                                                            0.2, 0.2]),
                         pre_nms_limit=6000,
                         images_per_gpu=1,
                         batch_size=1)([RPN_class, RPN_box, anchors])


@pytest.fixture
def detection_target_layer(proposal_layer, ground_truth):
    class_ids, boxes, masks = ground_truth
    target_layer = DetectionTargetLayer(images_per_gpu=1,
                                        mask_shape=[28, 28],
                                        train_ROIs_per_image=1,
                                        ROI_positive_ratio=0.33,
                                        bounding_box_std_dev=np.array([
                                            0.1, 0.1, 0.2, 0.2]),
                                        use_mini_mask=False,
                                        batch_size=1)
    return target_layer([proposal_layer, class_ids, boxes, masks])


@pytest.mark.parametrize('ROI_shape', [(3,)])
def test_proposal_layer(proposal_layer, ROI_shape):
    num_coordinates = 4
    assert proposal_layer.shape[2] == num_coordinates
    assert K.shape(proposal_layer).shape, ROI_shape


# @pytest.mark.parametrize('shapes', [[(3,), (2,), (3,), (4,)]])
# def test_detection_target_layer(detection_target_layer, shapes):
#     ROIs, target_class, target_box, target_mask = detection_target_layer
#     mask_shape = (28, 28)
#     results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
#                      K.shape(target_box).shape, K.shape(target_mask).shape]
#     assert shapes == results_shape
#     assert target_mask.shape[-2:] == mask_shape
#     assert ROIs.shape[2] == target_box.shape[2] == 4


@pytest.mark.parametrize('shape', [(1, 100, 6)])
def test_detection_layer(proposal_layer, FPN_classifier, shape):
    mrcnn_class, mrcnn_bounding_box = FPN_classifier
    detections = DetectionLayer(batch_size=1,
                                bounding_box_std_dev=np.array([0.1, 0.1,
                                                               0.2, 0.2]),
                                images_per_gpu=1,
                                detection_max_instances=100,
                                detection_min_confidence=0.7,
                                detection_nms_threshold=0.3,
                                image_shape=[128, 128, 3],
                                window=[0., 0., 128., 128.]
                                )([proposal_layer, mrcnn_class,
                                   mrcnn_bounding_box])
    assert detections.shape == shape


@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align(proposal_layer, feature_maps, shape):
    ROI_align = PyramidROIAlign([7, 7])([proposal_layer, shape] + feature_maps)
    assert K.int_shape(ROI_align) == (1, 2000, 7, 7, 256)
