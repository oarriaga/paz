from efficientdet import EFFICIENTDETD0
from utils import raw_images, preprocess_images, save_file
from efficientdet_postprocess import efficientdet_postprocess


if __name__ == "__main__":

    model = EFFICIENTDETD0()
    image_size = (raw_images.shape[0], model.image_size,
                  model.image_size, raw_images.shape[-1])
    input_image, image_scales = preprocess_images(raw_images, image_size)
    outputs = model(input_image)
    image, detections = efficientdet_postprocess(
        model, outputs, image_scales, raw_images)
    print(detections)
    save_file('paz_postprocess.jpg', image)
    print('task completed')
