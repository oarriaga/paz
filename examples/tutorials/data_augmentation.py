from paz.pipelines import AugmentImage
from paz.backend.image import show_image, load_image
import paz.processors as pr


# Testing default augmentation pipeline.
augment_image = AugmentImage()
for _ in range(5):
    image = load_image('image.jpg')
    image = augment_image(image)
    show_image(image)


# Putting something new somewhere.
augment_image.insert(0, pr.LoadImage())
for _ in range(5):
    image = augment_image('image.jpg')
    show_image(image)


# Adding new processor at the end to have a single function.
augment_image.add(pr.ShowImage())
for _ in range(5):
    image = augment_image('image.jpg')
