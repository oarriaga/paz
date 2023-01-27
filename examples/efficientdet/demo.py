from paz.processors.image import LoadImage
from paz.backend.image import show_image
from detection import EFFICIENTDETD0COCO

IMAGE_PATH = ('/home/manummk95/Desktop/efficientdet_BKP/'
              'paz/examples/efficientdet/img.jpg')
image = LoadImage()(IMAGE_PATH)
detect = EFFICIENTDETD0COCO()
detections = detect(image)
show_image(detections['image'])
