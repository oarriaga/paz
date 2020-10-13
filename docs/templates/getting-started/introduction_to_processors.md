# Introduction to processors

PAZ allows us to easily create preprocessing, data-augmentation and post-processing pipelines.

In the example below we show how to create a simple data-augmentation pipeline:

``` python
from paz.abstract import SequentialProcessor
from paz import processors as pr

augment_image = SequentialProcessor()
augment_image.add(pr.RandomContrast())
augment_image.add(pr.RandomBrightness())
augment_image.add(pr.RandomSaturation())
augment_image.add(pr.RandomHue())
```

The final pipeline (``augment_image``) behaves as a Python function:

``` python
new_image = augment_image(image)
```

There exists plenty default ``pipelines`` already built in PAZ. For more information please consult ``paz.pipelines``.


Pipelines are built from ``paz.processors``. There are plenty of processors implemented in PAZ; however, one can easily build a custom processor by inheriting from ``paz.abstract.Processor``.

In the example below we show how to build a ``processor`` for normalizing an image to a range from 0 to 1.

``` python
from paz.abstract import Processor

class NormalizeImage(Processor):
    """Normalize image by diving all values by 255.0.
    """
    def __init__(self):
        super(NormalizeImage, self).__init__()

    def call(self, image):
        return image / 255.0
```

We can now use our processor to create a pipeline for loading an image and normalizing it:

``` python
from paz.abstract import SequentialProcessor
from paz.processors import LoadImage

preprocess_image = SequentialProcessor()
preprocess_image.add(LoadImage())
preprocess_image.add(NormalizeImage())
```

We can now use our new function/pipeline to load and normalize an image:

``` python
image = preprocess_image('images/cat.jpg')
```

## Why the name ``Processor``?
Originally PAZ was only meant for pre-processing pipelines that included data-augmentation, normalization, etc. However, I found out that we could use the same API for post-processing; therefore, I thought at the time that ``Processor`` would be adequate to describe the capacity of both pre-processing and post-processing.
Names that I also thought could have worked were: ``Function``, ``Functor`` but I didn't want to use those since I thought they would be more confusing. Similarly, in Keras this abstraction is interpreted as a ``Layer`` but here I don't think that abstraction is adequate.
A layer of computation maybe? So after having this thoughts swirling around I decided to go with ``Processor`` and be explicit about my mental jugglery hoping that this name doesn't cause much mental overhead in the future.
