from paz.pipelines import AugmentImage
from paz.backend.image import show_image, load_image


for _ in range(10):
    image = load_image('image.jpg')
    augment_image = AugmentImage()
    image = augment_image(image)
    show_image(image)
