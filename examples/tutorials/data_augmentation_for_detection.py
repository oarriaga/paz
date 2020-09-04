# This script explains the basic functionality of ``SequentialProcessors`` for
# data augmentation in an object-detection task.

import os
import numpy as np
from paz.abstract import SequentialProcessor
from paz.backend.image import show_image, load_image
import paz.processors as pr
from paz.models.detection.utils import create_prior_boxes
from paz.backend.image import convert_color_space
from tensorflow.keras.utils import get_file

# let's download a test image and put it inside our PAZ directory
IMAGE_URL = ('https://github.com/oarriaga/altamira-data/releases/download'
             '/v0.9/test_image_detection.png')
image_filename = os.path.basename(IMAGE_URL)
image_fullpath = get_file(image_filename, IMAGE_URL, cache_subdir='paz/data')

'''
# We construct a pre-processing pipeline for **ONLY** our image:
preprocess_image = SequentialProcessor()
preprocess_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
preprocess_image.add(pr.ResizeImage((300, 300)))
preprocess_image.add(pr.CastImage(float))
preprocess_image.add(pr.SubtractMeanImage(pr.BGR_IMAGENET_MEAN))

# Let's see who it works:
image = load_image(image_fullpath)
image = preprocess_image(image)
show_image(image.astype('uint8'))


# Now let's build data augmentation for our image:
augment_image = SequentialProcessor()
augment_image.add(pr.RandomContrast())
augment_image.add(pr.RandomBrightness())
augment_image.add(pr.RandomSaturation())
augment_image.add(pr.RandomHue())

# Let's see who it works:
for _ in range(10):
    image = load_image(image_fullpath)
    image = augment_image(image)
    show_image(image.astype('uint8'))


# Let's put together our preprocessing and our augmentation for our image:
augment_image.add(preprocess_image)
for _ in range(10):
    image = load_image(image_fullpath)
    image = augment_image(image)
    show_image(image.astype('uint8'))
'''

# Now lets' work out the boxes part.
# Let's first build our labels:
# Keep in mind that the origin of our images is located at the **top-left**.

# The x_min, y_min are the **noramlized** coordinates
# of **top-left** bounding-box corner.
height, width = load_image(image_fullpath).shape[:2]
x_min_human, y_min_human = 200 / width, 60 / height
x_min_horse, y_min_horse = 100 / width, 90 / height

# The x_max, y_max are the **normalized** coordinates
# of **bottom-right** bounding-box corner.
x_max_human, y_max_human = 300 / width, 200 / height
x_max_horse, y_max_horse = 400 / width, 300 / height

# Our image has 1 + 2 classes. The **first** class is the background-class.
# The other 2 classes correspond to each object i.e. person (human), horse.
num_classes = 3
background_class, human_class, horse_class = 0, 1, 2
class_names = ['background', 'human', 'horse']

box_data = np.array(
    [[x_min_human, y_min_human, x_max_human, y_max_human, human_class],
     [x_min_horse, y_min_horse, x_max_horse, y_max_horse, horse_class]])

# Let's visualize our boxes!

# first we transform our numpy array into our built-in ``Box2D`` messages
to_boxes2D = pr.ToBoxes2D(class_names)
denormalize = pr.DenormalizeBoxes2D()

boxes2D = to_boxes2D(box_data)
image = load_image(image_fullpath)
boxes2D = denormalize(image, boxes2D)
# then we load, draw and show our image:
draw_boxes2D = pr.DrawBoxes2D(class_names)
show_image(draw_boxes2D(image, boxes2D))

# As you see were not able to put everything as a ``SequentialProcessor``
# This is because we are dealing with 2 inputs: ``box_data`` and ``image``.
# We can join them into a single processor using ``pr.ControlMap`` wrap.
# ``pr.ControlMap`` allows you to select which arguments (``intro_indices``)
# are passed to your processor, and also where you should put the output
# of your processor (``outro_indices``).
draw_boxes = SequentialProcessor()
draw_boxes.add(pr.ControlMap(to_boxes2D, intro_indices=[1], outro_indices=[1]))
draw_boxes.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
draw_boxes.add(pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1]))
# draw_boxes.add(pr.DrawBoxes2D(class_names))
# draw_boxes.add(pr.ShowImage())

# now you have everything in a single packed function that loads and draws!
a = draw_boxes(image_fullpath, box_data)

'''
# Let's now pre-process our boxes:

# Some object detectors create **positive boxes** by matching the ground
# truths with a set of **prior boxes**. Let's use for our case the prior
# boxes from the SSD300 detector
prior_boxes = create_prior_boxes()

# We begin by pre-processing our boxes
preprocess_boxes = SequentialProcessor()
preprocess_boxes.add(pr.MatchBoxes(prior_boxes))
preprocess_boxes.add(pr.EncodeBoxes(prior_boxes))
preprocess_boxes.add(pr.BoxClassToOneHotVector(num_classes))

draw_boxes.insert(0, pr.ControlMap(preprocess_boxes, [1], [1]))
draw_boxes.get_processor('ControlMap-ToBoxes2D').processor.one_hot_encoded = True
draw_boxes(image_fullpath, box_data)

def undo_preprocessing(image):
    image = (image + pr.BGR_IMAGENET_MEAN).astype('uint8')
    image = convert_color_space(image, pr.BGR2RGB)
    return image
'''
