from detection import EFFICIENTPOSEALINEMOD
from paz.backend.image import load_image, show_image

IMAGE_PATH = ('/home/manummk95/Desktop/ybkscht_efficientpose/EfficientPose/'
              'Datasets/Linemod_preprocessed/data/02/rgb/0000.png')


detect = EFFICIENTPOSEALINEMOD(score_thresh=0.90, nms_thresh=0.45)
image = load_image(IMAGE_PATH)
detections = detect(image)
show_image(detections['image'])
