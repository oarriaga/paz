import os
from tensorflow.keras.utils import get_file
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image, show_image


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.10/multi_person_test_pose.png')
filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)

detect = HigherHRNetHumanPose2D()
inferences = detect(image)

image = inferences['image']
show_image(image)
