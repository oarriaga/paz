import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from cityscapes import CityScapes
from paz import processors as pr
from processors import Round, MasksToColors
from model import UNET_VGG16


label_path = '/home/octavio/Downloads/dummy/gtFine/'
image_path = '/home/octavio/Downloads/dummy/RGB_images/leftImg8bit/'
data_manager = CityScapes(image_path, label_path, 'test')
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
        self.add(pr.ShowImage())


num_classes = len(data_manager.class_names)
input_shape = (128, 128, 3)
activation = 'softmax'
freeze = True
model = UNET_VGG16(num_classes, input_shape, 'imagenet', freeze, activation)
post_process = PostprocessSegmentation(model)
model.load_weights('experiments/model.tf')

for sample in data:
    image = post_process(sample)
