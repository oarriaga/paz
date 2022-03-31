import numpy as np
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
from paz.backend.image import write_image

from paz.pipelines import PIX2POSEPowerDrill
from paz.backend.camera import VideoPlayer


def approximate_intrinsics(image):
    image_size = image.shape[0:2]
    focal_length = image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
    camera_intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])
    return camera_intrinsics


# image = load_image('images/test_image2.jpg')
camera = Camera(device_id=4)
camera.start()
image = camera.read()
camera.stop()


camera.intrinsics = approximate_intrinsics(image)
camera.distortion = np.zeros((4))

pipeline = PIX2POSEPowerDrill(camera, offsets=[0.25, 0.25], epsilon=0.015)
"""
inferences = pipeline(image)
predicted_image = inferences['image']
show_image(predicted_image)
write_image('images/predicted_power_drill.png', predicted_image)
"""
player = VideoPlayer((640*3, 480*3), pipeline, camera)
# player = VideoPlayer((640, 480), pipeline, camera)
player.run()
