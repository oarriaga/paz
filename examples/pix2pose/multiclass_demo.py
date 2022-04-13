from paz.backend.image import show_image, load_image, resize_image
from paz.backend.camera import Camera
from pipelines import PIX2Tools6D


image = load_image('images/unit_test.png')
# image = load_image('images/group_photo_onefourth.png')
print(image.shape)
camera = Camera()
camera.intrinsics_from_HFOV(70, image.shape)
pipeline = PIX2Tools6D(camera)

inferences = pipeline(image)
predicted_image = inferences['image']
# H, W = predicted_image.shape[:2]
# predicted_image = resize_image(predicted_image, (W * 3, H * 3))
show_image(predicted_image)
