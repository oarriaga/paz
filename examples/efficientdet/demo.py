from tensorflow.keras.utils import get_file
from paz.backend.image import show_image, write_image
from paz.datasets import get_class_names
from paz.processors.image import LoadImage
from detection import DetectSingleShotEfficientDet, get_class_name_efficientdet
from efficientdet import EFFICIENTDETD0
from detection import efficientdet_preprocess, efficientdet_postprocess

IMAGE_PATH = ('/home/manummk95/Desktop/efficientdet_BKP/paz/examples/efficientdet/img.jpg')


if __name__ == "__main__":
    raw_image = LoadImage()(IMAGE_PATH)
    model = EFFICIENTDETD0(base_weights='COCO', head_weights='COCO')
    # image_size = model.input_shape[1]
    # input_image, image_scales = efficientdet_preprocess(raw_image, image_size)

    # outputs = model(input_image)
    
    # image, detections = efficientdet_postprocess(
    #     model, outputs, image_scales, raw_image)

    detect = DetectSingleShotEfficientDet(model, get_class_name_efficientdet('COCO'), 0.8, 0.4)
    detections = detect(raw_image)
    show_image(detections['image'])
    write_image('detections.png', detections['image'])
    print('task completed')