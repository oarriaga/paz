from paz.backend.image import show_image, write_image
from paz.processors.image import LoadImage
from detection import DetectSingleShotEfficientDet, get_class_name_efficientdet
from efficientdet import EFFICIENTDETD0

IMAGE_PATH = ('/home/manummk95/Desktop/efficientdet_BKP/paz/examples/'
              'efficientdet/img.jpg')

if __name__ == "__main__":
    raw_image = LoadImage()(IMAGE_PATH)
    model = EFFICIENTDETD0(base_weights='COCO', head_weights='COCO')
    detect = DetectSingleShotEfficientDet(
        model, get_class_name_efficientdet('COCO'), 0.8, 0.4)
    detections = detect(raw_image)
    show_image(detections['image'])
    write_image('detections.png', detections['image'])
    print('task completed')
