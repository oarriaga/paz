from paz.backend.image import show_image, load_image, resize_image
from paz.backend.camera import Camera
from paz.pipelines import PIX2YCBTools6D

image = load_image('images/unit_test.png')
# image = load_image('images/group_photo_onefourth.png')
print(image.shape)
camera = Camera()
camera.intrinsics_from_HFOV(55, image.shape)
pipeline = PIX2YCBTools6D(camera)

inferences = pipeline(image)
predicted_image = inferences['image']
H, W = predicted_image.shape[:2]
predicted_image = resize_image(predicted_image, (W * 3, H * 3))
show_image(predicted_image)

"""
from paz.backend.camera import VideoPlayer
camera = Camera(4)
camera.intrinsics_from_HFOV(55)
pipeline = PIX2YCBTools6D(camera, resize=False)
player = VideoPlayer((640 * 3, 480 * 3), pipeline, camera)
player.run()
"""
