# Image augmentation

This tutorial explains the basic functionality of ``SequentialProcessors`` for data augmentation in a classification task.

You can find the complete script of this tutorial [here](paz/examples/tutorials/image_augmentation.py)

``` python
import os
from paz.abstract import SequentialProcessor
from paz.backend.image import show_image, load_image
import paz.processors as pr
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
```

Let's download a test image and put it inside our PAZ directory
``` python
IMAGE_URL = ('https://github.com/oarriaga/altamira-data/releases/download'
             '/v0.9/image_augmentation.png')
image_filename = os.path.basename(IMAGE_URL)
image_fullpath = get_file(image_filename, IMAGE_URL, cache_subdir='paz/data')
```

We construct a data augmentation pipeline using the built-in PAZ processors:
``` python
augment = SequentialProcessor()
augment.add(pr.RandomContrast())
augment.add(pr.RandomBrightness())
augment.add(pr.RandomSaturation())
```

We can now apply our pipeline as a normal function:
``` python
for _ in range(5):
    image = load_image(image_fullpath)
    # use it as a normal function
    image = augment(image)
    show_image(image)
```

We can add to our sequential pipeline other function anywhere i.e. arg 0:
``` python
augment.insert(0, pr.LoadImage())
for _ in range(5):
    # now we don't load the image every time.
    image = augment(image_fullpath)
    show_image(image)
```

Adding new processor at the end to have a single function.
``` python
augment.add(pr.ShowImage())
for _ in range(5):
    # everything compressed into a single function
    image = augment(image_fullpath)
```

We can also pop the last processor added.
``` python
augment.pop()
```

We now create another processor for geometric augmentation.
**NOTE**: We can instantiate a new ``SequentialProcessor`` using a list of processors
``` python
transform = SequentialProcessor([pr.RandomRotation(), pr.RandomTranslation()])
```

You should start getting now transformations similar to these ones:

<p align="center">
   <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/examples_of_image_augmentation.png" width="800">
</p>



We can call both of our processors separately:
``` python
for _ in range(5):
    image = transform(augment(image_fullpath))
    show_image(image)
```

But since processors are just functions we can simply add it as a processor:
``` python
augment.add(transform)
for _ in range(5):
    image = augment(image_fullpath)
    show_image(image)
```

We can also use the Keras ``ImageDataGenerator``:
``` python
generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
```

We can add it by using our processor/wrapper ``ImageDataProcessor``:
``` python
augment = SequentialProcessor()
augment.add(pr.LoadImage())
augment.add(pr.ImageDataProcessor(generator))
augment.add(pr.ShowImage())
for _ in range(5):
    image = augment(image_fullpath)
```
