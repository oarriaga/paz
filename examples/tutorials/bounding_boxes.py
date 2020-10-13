import os
import numpy as np
from tensorflow.keras.utils import get_file

import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.backend.image import load_image

# let's download a test image and put it inside our PAZ directory
IMAGE_URL = ('https://github.com/oarriaga/altamira-data/releases/download'
             '/v0.9/object_detection_augmentation.png')
filename = os.path.basename(IMAGE_URL)
image_fullpath = get_file(filename, IMAGE_URL, cache_subdir='paz/tutorials')


# Boxes

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

# Let's create a simple visualization pipeline.
# For an explanation of what control-map is doing please check our tutorial at:
# paz/examples/tutorials/controlmap_processor.py
draw_boxes = SequentialProcessor()
draw_boxes.add(pr.ControlMap(pr.ToBoxes2D(class_names), [1], [1]))
draw_boxes.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
draw_boxes.add(pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}))
draw_boxes.add(pr.DrawBoxes2D(class_names))
draw_boxes.add(pr.ShowImage())


# We can now look at our boxes!
draw_boxes(image_fullpath, box_data)
