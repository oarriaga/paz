"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Input, Add, Conv2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model

from mask_rcnn.utils import log, mold_image, compose_image_meta
from mask_rcnn.utils import resnet_graph, build_rpn_model
from mask_rcnn.utils import norm_boxes, generate_pyramid_anchors
from mask_rcnn.utils import denorm_boxes, unmold_mask, resize_image
from mask_rcnn.utils import compute_backbone_shapes
tf.compat.v1.disable_eager_execution()


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    # Arguments:
        config: Instance of basic model configurations
        model_dir: Directory to save training logs and weights
    """

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.TRAIN_BN = config.TRAIN_BN
        self.IMAGE_SHAPE = config.IMAGE_SHAPE
        self.BACKBONE = config.BACKBONE
        self.TOP_DOWN_PYRAMID_SIZE = config.TOP_DOWN_PYRAMID_SIZE
        self.keras_model = self.build()

    def build(self):
        H, W = self.IMAGE_SHAPE[:2]
        if H / 2**6 != int(H / 2**6) or W / 2**6 != int(W / 2**6):
            raise Exception('Image size must be dividable by 2 atleast'
                            '6 times')

        input_image = Input(shape=[None, None, self.IMAGE_SHAPE[2]],
                            name='input_image')
        # input_image_meta = Input(shape=[self.config.IMAGE_META_SIZE],
        #                             name="input_image_meta")
        # if mode == "training":
        #     # RPN GT
        #     input_rpn_match = Input(
        #         shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        #     input_rpn_bbox = Input(
        #         shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        #     # Detection GT (class IDs, bounding boxes, and masks)
        #     # 1. GT Class IDs (zero padded)
        #     input_gt_class_ids = Input(
        #         shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        #     # 2. GT Boxes in pixels (zero padded)
        #     # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        #     input_gt_boxes = Input(
        #         shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        #     # Normalize coordinates
        #     gt_boxes = Lambda(lambda x: norm_boxes_graph(
        #         x, K.shape(input_image)[1:3]))(input_gt_boxes)
        #     # 3. GT Masks (zero padded)
        #     # [batch, height, width, MAX_GT_INSTANCES]
        #     if config.USE_MINI_MASK:
        #         input_gt_masks = Input(
        #             shape=[config.MINI_MASK_SHAPE[0],
        #                    config.MINI_MASK_SHAPE[1], None],
        #             name="input_gt_masks", dtype=bool)
        #     else:
        #         input_gt_masks = Input(
        #             shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
        #             name="input_gt_masks", dtype=bool)

        if callable(self.BACKBONE):
            _, C2, C3, C4, C5 = self.BACKBONE(input_image, stage5=True,
                                              train_bn=self.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, self.BACKBONE,
                                             stage5=True,
                                             train_bn=self.TRAIN_BN)

        P5 = Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = Add(name='fpn_p4add')([
            UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5),
            Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = Add(name='fpn_p3add')([
            UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4),
            Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                   name='fpn_c3p3')(C3)])
        P2 = Add(name='fpn_p2add')([
            UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3),
            Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p2')(P2)
        P3 = Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p3')(P3)
        P4 = Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p4')(P4)
        P5 = Conv2D(self.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p5')(P5)
        P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

        model = Model([input_image], [P2, P3, P4, P5, P6])

        # Anchors
        # if mode == "training":
        #     anchors = self.get_anchors(config.IMAGE_SHAPE)
        #     # Duplicate across the batch dimension because Keras requires it
        #     # TODO: can this be optimized to avoid duplicating the anchors?
        #     anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            
        #     #anchor_layer = AnchorsLayer(name='anchors')
        #     anchors = Layer(name='anchors')(anchors)
        #     # A hack to get around Keras's bad support for constants
        #     #anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        # else:
        #     anchors = input_anchors

        # if mode == "training":
        #     # Class ID mask to mark class IDs supported by the dataset the image
        #     # came from.
        #     active_class_ids = Lambda(
        #         lambda x: parse_image_meta_graph(x)["active_class_ids"]
        #         )(input_image_meta)

        #     if not config.USE_RPN_ROIS:
        #         # Ignore predicted ROIs and use ROIs provided as an input.
        #         input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
        #                               name="input_roi", dtype=np.int32)
        #         # Normalize coordinates
        #         target_rois = Lambda(lambda x: norm_boxes_graph(
        #             x, K.shape(input_image)[1:3]))(input_rois)
        #     else:
        #         target_rois = rpn_rois

        #     # Generate detection targets
        #     # Subsamples proposals and generates target outputs for training
        #     # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        #     # padded. Equally, returned rois and targets are zero padded.
        #     rois, target_class_ids, target_bbox, target_mask =\
        #         DetectionTargetLayer(config, name="proposal_targets")([
        #             target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        #     # Network Heads
        #     # TODO: verify that this handles zero padded ROIs
        #     mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        #         fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
        #                              config.POOL_SIZE, config.NUM_CLASSES,
        #                              mode=mode, train_bn=config.TRAIN_BN,
        #                              fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        #     mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
        #                                       input_image_meta,
        #                                       config.MASK_POOL_SIZE,
        #                                       config.NUM_CLASSES,
        #                                       train_bn=config.TRAIN_BN)

        #     # TODO: clean up (use tf.identify if necessary)
        #     output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

        #     # Losses
        #     RPN_output = [rpn_class_logits, rpn_bbox]
        #     target = [target_class_ids, target_bbox, target_mask]
        #     predictions = [mrcnn_class_logits, mrcnn_bbox, mrcnn_mask]
        #     loss = Loss(RPN_output, target, predictions, active_class_ids)
        #     loss = loss.compute_loss()
        #     rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        #         [input_rpn_match, rpn_class_logits])
        #     rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        #         [input_rpn_bbox, input_rpn_match, rpn_bbox])
        #     class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        #         [target_class_ids, mrcnn_class_logits, active_class_ids])
        #     bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        #         [target_bbox, target_class_ids, mrcnn_bbox])
        #     mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        #         [target_mask, target_class_ids, mrcnn_mask])

        #     # Model
        #     inputs = [input_image, input_image_meta,
        #               input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         inputs.append(input_rois)
        #     outputs = [rpn_class_logits, rpn_class, rpn_bbox,
        #                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
        #                rpn_rois, output_rois,
        #                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        #     model = Model(inputs, outputs, name='mask_rcnn')
        return model

    def RPN(self, rpn_feature_maps):
        rpn = build_rpn_model(self.config.RPN_ANCHOR_STRIDE,
                              len(self.config.RPN_ANCHOR_RATIOS),
                              self.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
        output_names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
        outputs = list(zip(*layer_outputs))
        return [Concatenate(axis=1, name=name)(list(output))
                for output, name in zip(outputs, output_names)]

    def find_last_checkpoint(self):
        directory_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        directory_names = filter(lambda f: f.startswith(key), directory_names)
        directory_names = sorted(directory_names)
        if not directory_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                'Could not find model directory {}'.format(self.model_dir))
        directory = os.path.join(self.model_dir, directory_names[-1])
        checkpoints = next(os.walk(directory))[2]
        checkpoints = filter(lambda f: f.startswith('mask_rcnn'), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, 'Could not find weights in {}'.format(directory))
        last_checkpoint = os.path.join(directory, checkpoints[-1])
        return last_checkpoint

    def get_imagenet_weights(self):
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/'\
                       'releases/download/v0.2/'\
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_file = get_file(filepath, WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_file

    def set_trainable(self, layer_regex, keras_model=None,
                      indent=0, verbose=1):
        if verbose > 0 and keras_model is None:
            log('Selecting layers to train')

        keras_model = keras_model or self.keras_model
        if hasattr(keras_model, 'inner_model'):
            layers = keras_model.inner_model.layers
        layers = keras_model.layers
        for layer in layers:
            if layer.__class__.__name__ == 'Model':
                print('In model: ', layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue
            if not layer.weights:
                continue
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def mold_inputs(self, images):
        """Mold inputs as expected by network

        # Arguments:
            images: List of image matrices [height,width,depth]

        # Returns:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas: [N, length of meta data]. Details about each image.
            windows: [N, (y_min, x_min, y_max, x_max)]
                     Image portion exculsing padding
        """
        molded_images, image_metas, windows = [], [], []
        for image in images:
            molded_image, window, scale, padding, crop = resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)

            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Unmolds the detections

        # Arguments:
            detections: [N, (y_min, x_min, y_max, x_max, class_id, score)]
            mrcnn_mask: [N, height, width, num_classes]
            original_image_shape: [H, W, num_channels] before resizing
            image_shape: [H, W, num_channels] After resizing
            window: [y_min, x_min, y_max, x_max]

        # Returns:
            boxes: [N, (y_min, x_min, y_max, x_max)]
                   Bounding boxes in pixels
            class_ids: Integer class IDs for each bounding box
            scores: Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
        """
        zero_index = np.where(detections[:, 4] == 0)[0]
        N = zero_index[0] if zero_index.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        window = norm_boxes(window, image_shape[:2])
        Wy_min, Wx_min, Wy_max, Wx_max = window
        shift = np.array([Wy_min, Wx_min, Wy_min, Wx_min])
        window_H = Wy_max - Wy_min
        window_W = Wx_max - Wx_min
        scale = np.array([window_H, window_W, window_H, window_W])
        boxes = np.divide(boxes - shift, scale)

        boxes = denorm_boxes(boxes, original_image_shape[:2])
        exclude_index = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_index.shape[0] > 0:
            boxes = np.delete(boxes, exclude_index, axis=0)
            class_ids = np.delete(class_ids, exclude_index, axis=0)
            scores = np.delete(scores, exclude_index, axis=0)
            masks = np.delete(masks, exclude_index, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        for index in range(N):
            full_mask = unmold_mask(masks[index], boxes[index],
                                    original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))
        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        # Arguments:
            images: List of images, potentially of different sizes.

        # Returns:
            rois: [N, (y_min, x_min, y_max, x_max)]
                  detection bounding boxes
            class_ids: N, int class IDs
            scores: N, float probability scores for the class IDs
            masks: [H, W, num_masks] instance binary masks
        """
        assert len(images) == self.config.BATCH_SIZE,\
            'len(images) must be equal to BATCH_SIZE'

        if verbose:
            log('Processing {} images'.format(len(images)))
            for image in images:
                log('image', image)

        molded_images, image_metas, windows = self.mold_inputs(images)
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                'After resizing, all images must have the same size'

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, 
                                  (self.config.BATCH_SIZE,) + anchors.shape)
        if verbose:
            log('molded_images', molded_images)
            log('image_metas', image_metas)
            log('anchors', anchors)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, anchors], verbose=0)

        results = []
        for index, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[index], mrcnn_mask[index],
                                       image.shape, molded_images[index].shape,
                                       windows[index])
            results.append({
                'rois': final_rois,
                'class_ids': final_class_ids,
                'scores': final_scores,
                'masks': final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size
        """
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, '_anchor_cache'):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            anchors = generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = anchors
            self._anchor_cache[tuple(image_shape)] = norm_boxes(
                                                    anchors, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.

        # Arguments:
            tensor: TensorFlow symbolic tensor.
            name: Name of ancestor tensor to find
            checked: A list of tensors that were already
                     searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        if len(checked) > 500:
            return None
        if isinstance(name, str):
            name = re.compile(name.replace('/', r'(\_\d+)*/'))
        for parent in tensor.op.inputs:
            if parent in checked:
                continue
            if bool(re.fullmatch(name, parent.name)):
                return parent
            checked.append(parent)
            ancestor = self.ancestor(parent, name, checked)
            if ancestor is not None:
                return ancestor
        return None

    def find_trainable_layer(self, layer):
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        layers = []
        for layer in self.keras_model.layers:
            layer = self.find_trainable_layer(layer)
            if layer.get_weights():
                layers.append(layer)
        return layers
