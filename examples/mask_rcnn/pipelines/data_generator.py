import numpy as np
import math
import tensorflow as tf

from paz.backend.image.image import cast_image

from mask_rcnn.backend.boxes import generate_pyramid_anchors, extract_boxes_from_mask
from mask_rcnn.backend.boxes import compute_rpn_bounding_box, compute_anchor_boxes_overlaps
from mask_rcnn.backend.boxes import compute_rpn_match

from mask_rcnn.backend.image import subtract_mean_image, generate_smaller_masks


def DataGenerator(dataset, backbone, image_shape, anchor_scales, batch_size, num_classes,
                  shuffle=True, pixel_mean=np.array([123.7, 116.8, 103.9])):
    """An iterable that returns images and corresponding target class ids,
        bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
        when multiprocessing=True.
        dataset: The Dataset object to pick data from
        shuffle: If True, shuffles the samples before every epoch
        augmentation: Optional. From paz pipeline for augmentation.
        inputs list:
        - images: [batch, H, W, C]
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bounding_box: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bounding_box deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.
        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bounding_box deltas,
            and masks.
        """
    b = 0
    image_index = -1
    image_ids = np.array([i for i in range(len(dataset.load_data()))])
    backbone_shapes = compute_backbone_shapes(backbone, image_shape)
    anchors = generate_pyramid_anchors(anchor_scales, [0.5, 1, 2],
                                       backbone_shapes, [4, 8, 16, 32, 64], 1)

    while True:
        try:

            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, image_id)

            rpn_match, rpn_bounding_box = build_rpn_targets(anchors, gt_class_ids, gt_boxes)

            if b == 0:
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bounding_box = np.zeros(
                    [batch_size, 256, 4], dtype=rpn_bounding_box.dtype)
                batch_images = np.zeros(
                    [batch_size, image_shape[0], image_shape[1], image_shape[2]], dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, 100), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, 100, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     100), dtype=bool)

                # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > 100:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), 100, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            image = cast_image(image, 'float32')

            # Add to batch
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bounding_box[b] = rpn_bounding_box
            batch_images[b] = subtract_mean_image(image, pixel_mean)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

            b += 1

            if b >= batch_size:

                inputs = [batch_images, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]

                zeros_array = np.zeros((batch_size, anchors.shape[0], 3))
                rpn_match_padded = np.concatenate((batch_rpn_match, zeros_array), axis=2)
                c = []
                for i in range(batch_size):
                    c.append(np.concatenate((batch_rpn_bounding_box[i], rpn_match_padded[i])))
                batch_rpn = np.stack(c, axis=0)

                outputs = [batch_rpn_match, batch_rpn]
                yield inputs, outputs

                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise


def compute_backbone_shapes(backbone, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(backbone):
        return compute_backbone_shapes(image_shape, image_shape)

    # Currently supports ResNet only
    assert backbone in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in [4, 8, 16, 32, 64]])


def load_image_gt(dataset, image_id, use_mini_mask=False, smaller_mask_shape=(28, 28)):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    augmentation: using paz library for augmentation

    # Returns:
    image: [height, width, 3]
    class_ids: [instance_count] Integer class IDs
    bounding_box: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    data = dataset.load_data()
    image = data[image_id]['image']
    mask = data[image_id]['masks']
    class_ids = data[image_id]['box_data'][:, -1]

    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]

    bounding_box = extract_boxes_from_mask(mask)

    box_reshape = bounding_box.copy()
    box_reshape[:, 0], box_reshape[:, 1], box_reshape[:, 2], box_reshape[:, 3] = \
        bounding_box[:, 1], bounding_box[:, 0], bounding_box[:, 3], bounding_box[:, 2]

    if use_mini_mask:
        mask = generate_smaller_masks(bounding_box.astype(np.int32), mask, smaller_mask_shape)

    return image, np.array(class_ids), box_reshape, mask.astype(np.bool)


def build_rpn_targets(anchors, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    # Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bounding_box: [N, (dy, dx, log(dh), log(dw))] Anchor bounding_box deltas.
    """
    overlaps, no_crowd_bool, gt_boxes = compute_anchor_boxes_overlaps(anchors, gt_class_ids, gt_boxes)

    rpn_match, anchor_iou_argmax = compute_rpn_match(anchors, overlaps, no_crowd_bool)

    rpn_bounding_box = compute_rpn_bounding_box(gt_boxes, rpn_match, anchors, anchor_iou_argmax)

    return rpn_match, rpn_bounding_box.astype(np.float32)
