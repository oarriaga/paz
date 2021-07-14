from efficientdet import EFFICIENTDET_D0
from misc import raw_images, preprocess_images
from postprocess import postprocess_paz


if __name__ == "__main__":

    model = EFFICIENTDET_D0()
    infer_on_image_size = model.image_size
    image_size = (raw_images.shape[0],
                  infer_on_image_size,
                  infer_on_image_size,
                  raw_images.shape[-1])
    input_image, image_scales = preprocess_images(raw_images, image_size)
    class_out, box_out = model(input_image)
    detections = postprocess_paz(class_out,
                                 box_out,
                                 image_scales,
                                 raw_images=raw_images,
                                 image_size=infer_on_image_size)
    print(detections)
    print('task completed')
