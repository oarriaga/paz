# Pipelines

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

The final ''pipeline'' behaves as any Python function:

``` python
new_image = augment_image(image)
```

There exists plenty default ''pipelines'' already built in PAZ. For more information please consult ''paz.pipelines''.


Pipelines are built from ''paz.processors''. There are plenty of processors implemented in PAZ; however, one can easily build a custom processor by inheriting from ''paz.abstract.Processor''.

In the example below we show how to build a ''processor'' for normalizing an image to a range from 0 to 1.

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
