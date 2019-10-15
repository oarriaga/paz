from paz.pipelines import DetectionAugmentation
from paz.models import SSD300
from paz.datasets import VOC
from paz.core import SequentialProcessor
from paz import processors as pr

size = 300
split = 'train'
learning_rate, momentum = 0.001, .9
epochs = 120
batch_size = 30

data_manager = VOC('VOCdevkit/')
data = data_manager.load_data()

num_classes = data_manager.num_classes
class_names = data_manager.class_names
model = SSD300(num_classes, base_weights='VGG', head_weights=None)
prior_boxes = model.prior_boxes

testor = SequentialProcessor()
testor.add(pr.LoadImage())
testor.add(pr.CastImageToFloat())
testor.add(pr.RandomContrast())
testor.add(pr.RandomBrightness())
testor.add(pr.CastImageToInts())
testor.add(pr.ConvertColor('BGR', to='HSV'))
testor.add(pr.CastImageToFloat())
testor.add(pr.RandomSaturation())
testor.add(pr.RandomHue())
testor.add(pr.CastImageToInts())
testor.add(pr.ConvertColor('HSV', to='BGR'))
testor.add(pr.RandomLightingNoise())
testor.add(pr.ToAbsoluteCoordinates())
testor.add(pr.Expand(mean=pr.BGR_IMAGENET_MEAN))
testor.add(pr.RandomSampleCrop())
testor.add(pr.HorizontalFlip())
testor.add(pr.ToPercentCoordinates())
testor.add(pr.Resize(shape=(size, size)))
testor.add(pr.CastImageToFloat())
testor.add(pr.MatchBoxes(prior_boxes, iou=.5))
testor.add(pr.EncodeBoxes(prior_boxes, variances=[.1, .2]))
testor.add(pr.ToOneHotVector(num_classes))
testor.add(pr.DecodeBoxes(prior_boxes, variances=[.1, .2]))
testor.add(pr.ToBoxes2D(class_names, True))
testor.add(pr.DenormalizeBoxes2D())
testor.add(pr.FilterClassBoxes2D(class_names[1:]))
testor.add(pr.CastImageToInts())
testor.add(pr.DrawBoxes2D(class_names))
testor.add(pr.ShowImage())


sample_arg = 2
sample = data[sample_arg]
boxes = testor(image=sample['image'], boxes=sample['boxes'].copy())


def go_batch():
    for sample_arg in range(batch_size):
        sample = data[sample_arg]
        testor(image=sample['image'], boxes=sample['boxes'].copy())
