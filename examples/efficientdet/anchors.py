import numpy as np
from paz.backend.boxes import to_center_form


def get_feature_sizes(image_size, max_level):
    """
    # Arguments
        max_level: Int, maximum level for features.
        image_size: Tuple, (Height, Width) of the input image.

    # Returns
        feat_sizes: List, feature sizes with height and width values
        for a given image size and max level.
    """
    feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feat_size = image_size
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
    return feat_sizes


class Anchors():
    """Generates anchors at different feature levels."""
    def __init__(self, min_level, max_level, num_scales, aspect_ratios,
                 anchor_scale, image_size):
        """
        # Arguments
            min_level: Int, minimum level for features.
            max_level: Int, maximum level for features.
            num_scales: Int, specifying the number of scales
            in the anchor boxes.
            aspect_ratios: List, specifying the aspect ratio of the
            default anchor boxes. Computed with k-mean on
            COCO dataset.
            anchor_scale: float number representing the scale
            of size of the base
            anchor to the feature stride 2^level. Or a list, one
            value per layer.
            image_size: Int, size of the input image.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales  # scale octave, how many P**2 available.
        self.aspect_ratios = aspect_ratios
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        if isinstance(anchor_scale, (list, tuple)):
            assert len(anchor_scale) == max_level - min_level + 1
            self.anchor_scales = anchor_scale
        else:
            self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)
        self.feat_sizes = get_feature_sizes(self.image_size, max_level)
        self.config = self._generate_configs()
        self.boxes = self._generate_boxes()

    def _generate_configs(self):
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(
                        ((feat_sizes[0]['height'] /
                          float(feat_sizes[level]['height']),
                         feat_sizes[0]['width'] /
                          float(feat_sizes[level]['width'])),
                         scale_octave / float(self.num_scales),
                         aspect,
                         self.anchor_scales[level - self.min_level]
                         ))
        return anchor_configs

    def get_aspect_ratio(self, aspect):
        if isinstance(aspect, list):
            aspect_x, aspect_y = aspect
        else:
            aspect_x = np.sqrt(aspect)
            aspect_y = 1 / aspect_x
        return aspect_x, aspect_y

    def get_box_coordinates(self, config):
        stride, octave_scale, aspect, anchor_scale = config
        base_anchor_x = anchor_scale * stride[1] * 2 ** octave_scale
        base_anchor_y = anchor_scale * stride[0] * 2 ** octave_scale
        aspect_x, aspect_y = self.get_aspect_ratio(aspect)
        anchor_x = base_anchor_x * aspect_x / 2.0
        anchor_y = base_anchor_y * aspect_y / 2.0
        x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
        y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
        center_x, center_y = np.meshgrid(x, y)
        center_x = center_x.reshape(-1)
        center_y = center_y.reshape(-1)
        return center_x, center_y, anchor_x, anchor_y

    def get_level_boxes(self, configs):
        boxes_level = []
        for config in configs:
            box_coordinates = self.get_box_coordinates(config)
            center_x, center_y, anchor_x, anchor_y = box_coordinates
            boxes = np.vstack((center_y - anchor_y, center_x - anchor_x,
                               center_y + anchor_y, center_x + anchor_x))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        return boxes_level

    def get_anchors(self):
        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = self.get_level_boxes(configs)
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchors = np.vstack(boxes_all)
        return anchors

    def _generate_boxes(self):
        anchor_boxes = self.get_anchors()
        anchor_boxes = anchor_boxes.astype('float32')
        return anchor_boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


def get_prior_boxes(min_level, max_level, num_scales, aspect_ratios,
                    anchor_scale, image_size):
    """Function to generate prior boxes.

    # Arguments
    min_level: Int, minimum level for features.
    max_level: Int, maximum level for features.
    num_scales: Int, specifying the number of scales in the anchor boxes.
    aspect_ratios: List, specifying the aspect ratio of the
    default anchor boxes. Computed with k-mean on COCO dataset.
    anchor_scale: float number representing the scale of size of the base
    anchor to the feature stride 2^level. Or a list, one value per layer.
    image_size: Int, size of the input image.

    # Returns
    prior_boxes: Numpy, Prior anchor boxes corresponding to the
    feature map size of each feature level.
    """
    prior_anchors = Anchors(min_level, max_level, num_scales,
                            aspect_ratios, anchor_scale, image_size)
    prior_boxes = prior_anchors.boxes
    a1, a2, a3, a4 = np.hsplit(prior_boxes, 4)
    prior_boxes = np.concatenate([a2, a1, a4, a3], axis=1)
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes
