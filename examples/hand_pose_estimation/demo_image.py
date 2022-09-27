import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image, show_image
from paz.applications import MinimalHandPoseEstimation


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.14/image_with_hand.png')
filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)

detect = MinimalHandPoseEstimation(right_hand=False)
inferences = detect(image)

image = inferences['image']
show_image(image)
