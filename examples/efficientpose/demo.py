from paz.backend.image import load_image, show_image
from pose import EFFICIENTPOSEALINEMOD

IMAGE_PATH = ('/home/manummk95/Desktop/ybkscht_efficientpose/EfficientPose/'
              'Datasets/Linemod_preprocessed/data/02/rgb/0000.png')


detect = EFFICIENTPOSEALINEMOD(score_thresh=0.60, nms_thresh=0.45,
                               show_boxes2D=False, show_poses6D=True)
image = load_image(IMAGE_PATH)
inferences = detect(image)
show_image(inferences['image'])
