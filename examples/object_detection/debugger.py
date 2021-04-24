import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


import numpy as np
from paz.models import SSD300
from paz.datasets import VOC
from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from detection import AugmentDetection
# from paz.pipelines import AugmentDetection


class ShowBoxes(Processor):
    def __init__(self, class_names, prior_boxes,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(ShowBoxes, self).__init__()
        self.deprocess_boxes = SequentialProcessor([
            pr.DecodeBoxes(prior_boxes, variances),
            pr.ToBoxes2D(class_names, True),
            pr.FilterClassBoxes2D(class_names[1:])])
        self.denormalize_boxes2D = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.show_image = pr.ShowImage()
        self.resize_image = pr.ResizeImage((600, 600))

    def call(self, image, boxes):
        image = self.resize_image(image)
        boxes2D = self.deprocess_boxes(boxes)
        boxes2D = self.denormalize_boxes2D(image, boxes2D)
        image = self.draw_boxes2D(image, boxes2D)
        image = (image + pr.BGR_IMAGENET_MEAN).astype(np.uint8)
        image = image[..., ::-1]
        self.show_image(image)
        return image, boxes2D


size = 300
split = 'train'
epochs = 120
batch_size = 30

data_manager = VOC('VOCdevkit/')
data = data_manager.load_data()

class_names = data_manager.class_names
# model = SSD300(base_weights='VGG', head_weights=None)
model = SSD300()
prior_boxes = model.prior_boxes

testor_encoder = AugmentDetection(prior_boxes)
testor_decoder = ShowBoxes(class_names, prior_boxes)
sample_arg = 0
for sample_arg in range(1000):
    sample = data[sample_arg]
    wrapped_outputs = testor_encoder(sample)
    print(wrapped_outputs['labels'])
    image = wrapped_outputs['inputs']['image']
    boxes = wrapped_outputs['labels']['boxes']
    image, boxes = testor_decoder(image, boxes)
