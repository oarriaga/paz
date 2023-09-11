import os
from paz.backend.camera import Camera
from scipy.optimize import least_squares
from tensorflow.keras.utils import get_file
from paz.applications import EstimateHumanPose
from paz.backend.image import load_image, show_image


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.17/multiple_persons_posing.png')

filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)
H, W = image.shape[:2]
camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=[H, W])
pipeline = EstimateHumanPose(least_squares, camera.intrinsics)
inference = pipeline(image)
show_image(inference['image'])
