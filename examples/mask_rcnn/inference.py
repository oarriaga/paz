from mask_rcnn.model import MaskRCNN
from mask_rcnn.config import Config
from mask_rcnn.utils import norm_boxes_graph
from mask_rcnn.inference_graph import InferenceGraph
from mask_rcnn.detection import ResizeImages, NormalizeImages
from mask_rcnn.detection import Detect, PostprocessInputs
from paz.abstract import SequentialProcessor


image_min_dim = 128
image_max_dim = 128
image_scale = 0
image_shape = [128, 128, 3]
anchor_ratios = (8, 16, 32, 64, 128)
images_per_gpu = 1


def test(images, weights_path):
    config = TestConfig()
    resize = SequentialProcessor([ResizeImages(image_min_dim, image_scale, image_max_dim)])
    molded_images, windows = resize(images)
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])

    base_model = MaskRCNN(model_dir='../../mask_rcnn', image_shape=image_shape, backbone="resnet101",
                          batch_size=1, images_per_gpu=1, rpn_anchor_scales=(8, 16, 32, 64, 128),
                          train_rois_per_image=32, num_classes=4, window=window)
    inference_model = InferenceGraph(model=base_model, config=config)
    base_model.keras_model = inference_model()
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(image_min_dim, image_scale, image_max_dim),
                                      NormalizeImages()])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, anchor_ratios, images_per_gpu, preprocess, postprocess)
    results = detect(images)
    return results
