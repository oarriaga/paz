# Control-map

In this tutorial we show you how to use one of our most useful processors ``ControlMap``.

You can find the complete script of this tutorial [here](https://github.com/oarriaga/paz/blob/master/examples/tutorials/controlmap_processor.py)

``` python
import os
import numpy as np
from paz.abstract import SequentialProcessor
from paz.backend.image import show_image, load_image
import paz.processors as pr
from tensorflow.keras.utils import get_file
```

Let's download a test image and put it inside our PAZ directory
``` python
IMAGE_URL = ('https://github.com/oarriaga/altamira-data/releases/download'
             '/v0.9/object_detection_augmentation.png')
image_filename = os.path.basename(IMAGE_URL)
image_fullpath = get_file(image_filename, IMAGE_URL, cache_subdir='paz/data')
```

The ``x_min, y_min`` are the **normalized** coordinates of **top-left** bounding-box corner.
``` python
H, W = load_image(image_fullpath).shape[:2]
```

The ``x_max, y_max`` are the **normalized** coordinates
``` python
class_names = ['background', 'human', 'horse']
box_data = np.array([[200 / W, 60 / H, 300 / W, 200 / H, 1],
                     [100 / W, 90 / H, 400 / W, 300 / H, 2]])
```

Let's visualize our boxes!
First we transform our numpy array into our built-in ``Box2D`` messages
``` python
to_boxes2D = pr.ToBoxes2D(class_names)
denormalize = pr.DenormalizeBoxes2D()
boxes2D = to_boxes2D(box_data)
image = load_image(image_fullpath)
boxes2D = denormalize(image, boxes2D)
draw_boxes2D = pr.DrawBoxes2D(class_names)
show_image(draw_boxes2D(image, boxes2D))
```

As you can see, we were not able to put everything as a ``SequentialProcessor``.
This is because we are dealing with 2 inputs: ``box_data`` and ``image``.
We can join them into a single processor using ``pr.ControlMap`` wrap.
``pr.ControlMap`` allows you to select which arguments (``intro_indices``) are passed to your processor, and also where you should put the output of your processor (``outro_indices``).
``` python
draw_boxes = SequentialProcessor()
draw_boxes.add(pr.ControlMap(to_boxes2D, intro_indices=[1], outro_indices=[1]))
draw_boxes.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
draw_boxes.add(pr.ControlMap(denormalize, [0, 1], [1], keep={0: 0}))
draw_boxes.add(pr.DrawBoxes2D(class_names))
draw_boxes.add(pr.ShowImage())
```

Now you have everything in a single packed function that loads and draws!
``` python
draw_boxes(image_fullpath, box_data)
```

Also note if one of your function is ``eating`` away one input that you wish to keep in your pipeline, you can use the ``keep`` dictionary to explicitly say which of your inputs you wish to keep and where it should be located.
This is represented respectively by the ``key`` and the ``value`` of the ``keep`` dictionary.
