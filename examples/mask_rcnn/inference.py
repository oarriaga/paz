from .model import MaskRCNN
from .config import Config
from .utils import norm_boxes_graph
from .inference_graph import InferenceGraph
from .detection import ResizeImages, NormalizeImages
from .detection import Detect, PostprocessInputs
from paz.abstract import SequentialProcessor


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def test(images, weights_path):
    config = TestConfig()
    resize = SequentialProcessor([ResizeImages(config)])
    molded_images, windows = resize(images)
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])
    config.WINDOW = window
    train_bn= config.TRAIN_BN
    image_shape= config.IMAGE_SHAPE
    backbone= config.BACKBONE
    top_down_pyramid_size= config.TOP_DOWN_PYRAMID_SIZE

    base_model = MaskRCNN(config=config, model_dir='../../mask_rcnn', train_bn=train_bn, image_shape=image_shape,
                          backbone=backbone, top_down_pyramid_size=top_down_pyramid_size)
    inference_model = InferenceGraph(model=base_model, config=config)
    base_model.keras_model = inference_model()
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(config),
                                      NormalizeImages(config)])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, config, preprocess, postprocess)
    results = detect(images)
    return results