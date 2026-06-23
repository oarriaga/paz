from keras import Model
from keras.layers import Input
from keras.utils import get_file

from paz.backend.anchors import build_anchors, build_translation_anchors
from ...detection.efficientdet.efficientnet import EFFICIENTNET
from ...detection.efficientdet.efficientdet_blocks import build_detector_head
from ...detection.efficientdet.efficientdet_blocks import EfficientNet_to_BiFPN
from ...detection.efficientdet.efficientdet_blocks import BiFPN
from .efficientpose_blocks import build_pose_estimator_head


WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.16/"  # fmt: skip

PHI_CONFIG = {
    0: dict(size=512, filters=64, cells=3, repeats=2, iterations=1,
            box_class=3, anchor_scale=4.0, fusion="fast", scaling=(1.0, 1.0, 0.8)),  # fmt: skip
    1: dict(size=640, filters=88, cells=4, repeats=2, iterations=1,
            box_class=3, anchor_scale=4.0, fusion="fast", scaling=(1.0, 1.0, 0.8)),  # fmt: skip
    2: dict(size=768, filters=112, cells=5, repeats=2, iterations=1,
            box_class=3, anchor_scale=4.0, fusion="fast", scaling=(1.1, 1.2, 0.7)),  # fmt: skip
    3: dict(size=896, filters=160, cells=6, repeats=3, iterations=2,
            box_class=4, anchor_scale=4.0, fusion="fast", scaling=(1.2, 1.4, 0.7)),  # fmt: skip
    4: dict(size=1024, filters=224, cells=7, repeats=3, iterations=2,
            box_class=4, anchor_scale=4.0, fusion="fast", scaling=(1.2, 1.4, 0.7)),  # fmt: skip
    5: dict(size=1280, filters=288, cells=7, repeats=3, iterations=2,
            box_class=4, anchor_scale=4.0, fusion="fast", scaling=(1.6, 2.2, 0.6)),  # fmt: skip
    6: dict(size=1280, filters=384, cells=8, repeats=4, iterations=3,
            box_class=5, anchor_scale=5.0, fusion="sum", scaling=(1.8, 2.6, 0.5)),  # fmt: skip
    7: dict(size=1536, filters=384, cells=8, repeats=4, iterations=3,
            box_class=5, anchor_scale=5.0, fusion="sum", scaling=(1.8, 2.6, 0.5)),  # fmt: skip
}


def EfficientPosePhi0(**kwargs):
    return build_efficientpose(0, **kwargs)


def EfficientPosePhi1(**kwargs):
    return build_efficientpose(1, **kwargs)


def EfficientPosePhi2(**kwargs):
    return build_efficientpose(2, **kwargs)


def EfficientPosePhi3(**kwargs):
    return build_efficientpose(3, **kwargs)


def EfficientPosePhi4(**kwargs):
    return build_efficientpose(4, **kwargs)


def EfficientPosePhi5(**kwargs):
    return build_efficientpose(5, **kwargs)


def EfficientPosePhi6(**kwargs):
    return build_efficientpose(6, **kwargs)


def EfficientPosePhi7(**kwargs):
    return build_efficientpose(7, **kwargs)


def build_efficientpose(phi, num_classes=8, base_weights="COCO",
                        head_weights="LinemodOccluded", return_base=False):
    config = PHI_CONFIG[phi]
    image = Input(shape=(config["size"], config["size"], 3), name="image")
    backbone = EFFICIENTNET(image, config["scaling"])
    model_name = f"EfficientPose-Phi{phi}"
    return EfficientPose(image, backbone, config, num_classes, base_weights,
                         head_weights, return_base, model_name)


def EfficientPose(image, backbone, config, num_classes, base_weights,
                  head_weights, return_base, model_name, num_scales=3,
                  aspect_ratios=(1.0, 2.0, 0.5), num_dims=4, num_anchors=9,
                  num_filters=64, num_pose_dims=3):
    validate_weights(base_weights, head_weights)
    branches, middles, skips = EfficientNet_to_BiFPN(backbone, config["filters"])  # fmt: skip
    for _ in range(config["cells"]):
        middles, skips = BiFPN(middles, skips, config["filters"], config["fusion"])  # fmt: skip
    outputs = middles
    if not return_base:
        outputs = build_heads(middles, config, num_classes, num_scales,
                              aspect_ratios, num_dims, num_anchors,
                              num_filters, num_pose_dims)
    model = Model(inputs=image, outputs=outputs, name=model_name)
    load_weights(model, model_name, base_weights, head_weights)
    image_shape = list(image.shape[1:3])
    anchor_args = (image_shape, branches, num_scales, aspect_ratios)
    model.prior_boxes = build_anchors(*anchor_args, config["anchor_scale"])
    model.translation_priors = build_translation_anchors(*anchor_args)
    return model


def build_heads(middles, config, num_classes, num_scales, aspect_ratios,
                num_dims, num_anchors, num_filters, num_pose_dims):
    detection_args = (middles, num_classes, num_dims, aspect_ratios,
                      num_scales, config["filters"], config["box_class"], None)
    detections = build_detector_head(*detection_args)
    pose_args = (middles, config["iterations"], config["repeats"],
                 num_anchors, num_filters, num_pose_dims)
    poses = build_pose_estimator_head(*pose_args)
    return [detections, poses]


def load_weights(model, model_name, base_weights, head_weights):
    if (base_weights is None) and (head_weights is None):
        return
    suffix = str(head_weights) + "_weights.hdf5"
    filename = "-".join([model_name, str(base_weights), suffix])
    weights_path = get_file(filename, WEIGHT_PATH + filename,
                            cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    model.load_weights(weights_path, by_name=head_weights is None)


def validate_weights(base_weights, head_weights):
    if base_weights not in ["COCO", None]:
        raise ValueError("Invalid base_weights:", base_weights)
    if head_weights not in ["LinemodOccluded", None]:
        raise ValueError("Invalid head_weights:", head_weights)
