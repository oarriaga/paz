import os
import numpy as np

from paz.abstract import ProcessingSequence, SequentialProcessor
from paz.backend.image import convert_color_space
from paz import processors as pr
from paz.models import SSD300
from paz.pipelines import AugmentDetection

# from hand_dataset import HandDataset
from open_images import OpenImagesV6

batch_size = 32
num_classes = 2
class_names = ['background', 'hand']


root_path = os.path.expanduser('~')
data_path = os.path.join(root_path, 'hand_dataset/hand_dataset/')
data_path = os.path.join(root_path, '/home/octavio/fiftyone/open-images-v6/')
data_manager = OpenImagesV6(data_path, pr.TRAIN, ['background', 'Human hand'])
data = data_manager.load_data()
model = SSD300(num_classes, base_weights='VGG',
               head_weights=None, trainable_base=False)


augmentator = AugmentDetection(model.prior_boxes, pr.TRAIN, num_classes)
sequence = ProcessingSequence(augmentator, batch_size, data)


class ShowBoxes(pr.Processor):
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
        image = (image + pr.BGR_IMAGENET_MEAN).astype(np.uint8)
        image = self.draw_boxes2D(image, boxes2D)
        image = convert_color_space(image, pr.BGR2RGB)
        self.show_image(image)
        return image, boxes2D


show_boxes = ShowBoxes(class_names, model.prior_boxes)

batch = sequence.__getitem__(65)
for sample_arg in range(batch_size):
    batch_images, batch_boxes = batch[0]['image'], batch[1]['boxes']
    image, boxes = batch_images[sample_arg], batch_boxes[sample_arg]
    show_boxes(image, boxes)
