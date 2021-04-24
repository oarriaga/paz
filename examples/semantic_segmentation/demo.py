import numpy as np
import tensorflow as tf
import argparse
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from paz.datasets import CityScapes
from paz import processors as pr
from paz.backend.image import resize_image, load_image, show_image
from processors import Round, MasksToColors
from processors import ResizeImageWithNearestNeighbors
from paz.models import UNET_VGG16


description = 'Training script for semantic segmentation'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--label_path', type=str, help='Path to labels')
parser.add_argument('--image_path', type=str, help='Path to images')
parser.add_argument('--weights_path', type=str, help='Path to weights')
args = parser.parse_args()

data_manager = CityScapes(args.image_path, args.label_path, 'test')
data = data_manager.load_data()


class PostprocessSegmentation(pr.SequentialProcessor):
    def __init__(self, model, colors=None):
        super(PostprocessSegmentation, self).__init__()
        self.add(pr.UnpackDictionary(['image_path']))
        self.add(pr.LoadImage())
        self.add(pr.ResizeImage(model.input_shape[1:3]))
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.add(pr.SubtractMeanImage(pr.BGR_IMAGENET_MEAN))
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(model))
        self.add(pr.Squeeze(0))
        self.add(Round())
        self.add(MasksToColors(model.output_shape[-1], colors))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))
        self.add(ResizeImageWithNearestNeighbors((1024, 512)))
        # self.add(pr.ShowImage())


num_classes = len(data_manager.class_names)
input_shape = (128, 128, 3)
model = UNET_VGG16(num_classes, input_shape, 'imagenet', activation='softmax')
post_process = PostprocessSegmentation(model)
model.load_weights(args.weights_path)

for sample in data:
    masks = post_process(sample)
    image = load_image(sample['image_path'])
    image = resize_image(image, (1024, 512))
    image_with_masks = ((0.6 * image) + (0.4 * masks)).astype("uint8")
    # image_and_masks = np.concatenate([image, masks], axis=1)
    show_image(image_with_masks)
