from paz.backend.image import show_image, write_image
from paz.processors.image import LoadImage
from detection import EFFICIENTDETD0COCO

IMAGE_PATH = ('/home/manummk95/Desktop/efficientdet_BKP/paz/examples/'
              'efficientdet/img.jpg')

if __name__ == "__main__":
    raw_image = LoadImage()(IMAGE_PATH)
    detect = EFFICIENTDETD0COCO()
    detections = detect(raw_image)
    show_image(detections['image'])
    write_image('detections.png', detections['image'])
    print('task completed')
