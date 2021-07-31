from efficientdet import EFFICIENTDETD0
from utils import raw_images, preprocess_images
from efficientdet_postprocess import efficientdet_postprocess


if __name__ == "__main__":

    model = EFFICIENTDETD0()
    infer_on_image_size = model.image_size
    image_size = (raw_images.shape[0], infer_on_image_size,
                  infer_on_image_size, raw_images.shape[-1])
    input_image, image_scales = preprocess_images(raw_images, image_size)
    class_out, box_out = model(input_image)
    detections = efficientdet_postprocess(
        class_out, box_out, infer_on_image_size, image_scales, raw_images)
    print(detections)
    print('task completed')
