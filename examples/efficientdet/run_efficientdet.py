from efficientdet_model import EfficientDet
from misc import raw_images, load_pretrained_weights, preprocess_images
from postprocess import postprocess_paz


if __name__ == "__main__":

    model = EfficientDet()
    model.build(raw_images.shape)
    print(model.summary())

    WEIGHT = '/media/deepan/externaldrive1/project_repos/' \
             'paz/examples/efficientdet/efficientdet-d0.h5'
    infer_on_image_size = 512
    model = load_pretrained_weights(model, WEIGHT)
    print('Successfully copied weights.')

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
