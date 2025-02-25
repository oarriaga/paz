import os
from jax import vmap
from paz import Input, Node, Model
import paz

image_path = Input("image_path")
x = Node(paz.image.load)(image_path)
x = Node(paz.image.scale, (0.5, 0.5))(x)
x = Node(paz.image.normalize)(x)
x = Node(paz.image.rgb_to_gray)(x)
x = Node(lambda x: (x * 255).astype("uint8"), name="deprocess")(x)
x = Node(paz.image.show)(x)
preprocess = Model([image_path], [x], "preprocess")

preprocess(os.path.join(os.path.expanduser("~"), "images/mars_volta.jpg"))

# preprocess_batch = vmap(Model(image, x))

# TODO
# Fix node name creation with lambda function.
# Have a nice pretty drawing for model pipeline.
# Build tests
# Test bounding box and image preprocessing i.e. two functions
# Build sequential abstraction
