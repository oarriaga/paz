import numpy as np
from paz.models import SSD300
from paz.datasets import VOC
from paz.abstract import Processor, SequentialProcessor
from paz.abstract import ProcessingSequence
from paz import processors as pr
from paz.pipelines import AugmentDetection
import cv2

class ShowBoxes(Processor):
    def __init__(self, class_names, prior_boxes, variances=[.1, .2]):
        super(ShowBoxes, self).__init__()
        self.deprocess_boxes = SequentialProcessor([
            pr.DecodeBoxes(prior_boxes, variances),
            pr.ToBoxes2D(class_names, True),
            pr.FilterClassBoxes2D(class_names[1:])])
        self.denormalize_boxes2D = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.show_image = pr.ShowImage()

    def call(self, image, boxes):
        boxes2D = self.deprocess_boxes(boxes)
        boxes2D = self.denormalize_boxes2D(image, boxes2D)
        image = self.draw_boxes2D(image, boxes2D)
        image = (image + pr.BGR_IMAGENET_MEAN).astype(np.uint8)
        image = image[..., ::-1]
        cv2.imwrite('image.png', image)
        #self.show_image(image)
        return image, boxes2D


size = 300
split = 'train'
epochs = 120
batch_size = 30

data_manager = VOC('VOCdevkit/')
data = data_manager.load_data()

class_names = data_manager.class_names
model = SSD300(base_weights='VGG', head_weights=None)
prior_boxes = model.prior_boxes

testor_encoder = AugmentDetection(prior_boxes)
testor_decoder = ShowBoxes(class_names, prior_boxes)
sample_arg = 0
for sample_arg in range(50):
    sample = data[sample_arg]
    if(type(sample) == str):
        print("The sample is a string")
    wrapped_outputs = testor_encoder(sample)
    #print(wrapped_outputs['labels'])
    image = wrapped_outputs['inputs']['image']
    boxes = wrapped_outputs['labels']['boxes']
    image, boxes = testor_decoder(image, boxes)

sequencer = ProcessingSequence(testor_encoder, 32, data)

for batch_arg in range(len(data)):
    sequencer.__getitem__(batch_arg)
