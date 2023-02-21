import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image, show_image
from paz.pipelines import EFFICIENTDETD0COCO

URL = ('https://github.com/oarriaga/altamira-data/releases/download/v0.16/'
       'image_with_multiple_objects.png')

filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)

detect = EFFICIENTDETD0COCO(score_thresh=0.60, nms_thresh=0.25)
detections = detect(image)
show_image(detections['image'])
