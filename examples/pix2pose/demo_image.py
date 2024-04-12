import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image, show_image
from paz.applications import PIX2YCBTools6D
from paz.backend.camera import Camera


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.9.1/image_with_YCB_objects.jpg')
filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)
camera = Camera()
camera.intrinsics_from_HFOV(55, image.shape)

detect = PIX2YCBTools6D(camera, offsets=[0.25, 0.25], epsilon=0.015)
inferences = detect(image)

image = inferences['image']
show_image(image)
