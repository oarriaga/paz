from paz.abstract import SequentialProcessor

from mask_rcnn.model.model import MaskRCNN, norm_all_boxes
from mask_rcnn.pipelines.detection import ResizeImages, NormalizeImages
from mask_rcnn.pipelines.detection import Detect, PostprocessInputs


def test(images, weights_path, ROIs_per_image, num_classes, batch_size,
         images_per_gpu, anchor_ratios, image_shape, min_image_scale):
    resize = SequentialProcessor([ResizeImages(image_shape[0], min_image_scale,
                                               image_shape[1])])
    molded_images, windows = resize([images])
    image_shape = molded_images[0].shape
    window = norm_all_boxes(windows[0], image_shape[:2])

    base_model = MaskRCNN(model_dir='../../mask_rcnn',
                          image_shape=image_shape,
                          backbone="resnet101",
                          batch_size=batch_size, images_per_gpu=images_per_gpu,
                          RPN_anchor_scales=anchor_ratios,
                          train_ROIs_per_image=ROIs_per_image,
                          num_classes=num_classes,
                          window=window)

    base_model.build_model(train=False)
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(), NormalizeImages()])
    postprocess = SequentialProcessor([PostprocessInputs()])

    detect = Detect(base_model, anchor_ratios, images_per_gpu, preprocess,
                    postprocess)
    results = detect([images])
    return results
