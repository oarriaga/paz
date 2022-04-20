import numpy as np
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera

# from paz.pipelines import PIX2POSEPowerDrill
from paz.backend.camera import VideoPlayer
from pipelines import PIX2POSEPowerDrill


image = load_image('images/test_image2.jpg')
camera = Camera()
camera.intrinsics_from_HFOV(70, image.shape)

pipeline = PIX2POSEPowerDrill(camera, offsets=[0.25, 0.25], epsilon=0.015)
inferences = pipeline(image)
predicted_image = inferences['image']
show_image(predicted_image)

"""
player = VideoPlayer((640*3, 480*3), pipeline, camera)
player.run()
"""
