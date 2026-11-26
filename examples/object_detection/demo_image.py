import paz

image = paz.image.load("photo_2.jpg")
paz.image.show(image)
image_resized = paz.image.resize_with_aspect_ratio(image, 300, 300)
paz.image.show(image_resized.astype("uint8"))
print(image_resized.shape)

model = paz.models.EFFICIENTDETD0()
print(model.input_shape)
print(model.output_shape)
