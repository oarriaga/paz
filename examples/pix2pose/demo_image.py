from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
from paz.pipelines import PIX2POSEPowerDrill


image = load_image('images/test_image2.jpg')
camera = Camera()
camera.intrinsics_from_HFOV(55, image.shape)

pipeline = PIX2POSEPowerDrill(camera, offsets=[0.25, 0.25], epsilon=0.015)
inferences = pipeline(image)
show_image(inferences['image'])


import glob
for image_path in glob.glob('images/test_0*.jpg'):
    print(image_path)
    image = load_image(image_path)
    inferences = pipeline(image)
    show_image(inferences['image'])
