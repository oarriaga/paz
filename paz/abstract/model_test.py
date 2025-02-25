from jax import vmap
from paz import Input, Node, Model
import paz

image_path = Input("image_path")
x = Node(paz.image.load)(image_path)
x = Node(paz.image.show)(x)
# x = Node(paz.image.rgb_to_gray, name="image")(y)
preprocess = Model([image_path], [x], "preprocess")
# preprocess_batch = vmap(Model(image, x))


"""
probs = paz.Input("probs")
names = Node(paz.lock(paz.classes.to_name, class_names))(classes)
postprocess = jax.vmap(Model(probs, [probs, names]))

batch_images = Input("batch_images")
x = Node(preprocess)(batch_images)
x = model(x)
x = Node(postprocess)(x)
x = Node(paz.draw.boxes)(x)
model = Model(batch_images, x)
"""
