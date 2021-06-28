from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from processors import MatchBoxes
import numpy as np
import os


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.

    # Arguments
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        if mean is None:
            self.add(pr.NormalizeImage())
        else:
            self.add(pr.SubtractMeanImage(mean))


class AugmentImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.

    # Arguments
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, bkg_paths, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentImage, self).__init__()
        # self.add(LoadImage(4))
        self.add(pr.ResizeImage(shape))
        self.add(pr.BlendRandomCroppedBackground(bkg_paths))
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation(0.7))
        self.add(pr.RandomHue())
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.

    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class DrawBoxData2D(Processor):
    def __init__(self, class_names, preprocess=None, colors=None):
        super(DrawBoxData2D, self).__init__()
        self.class_names, self.colors = class_names, colors
        self.to_boxes2D = pr.ToBoxes2D(self.class_names)
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names, self.colors)
        self.preprocess = preprocess

    def call(self, image, boxes):
        if self.preprocess is not None:
            image, boxes = self.preprocess(image, boxes)
            boxes = boxes.astype('int')
        boxes = self.to_boxes2D(boxes)
        print(boxes)
        image = self.draw_boxes2D(image, boxes)
        return image, boxes


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


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.

    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, bkg_paths, split=pr.TRAIN, num_classes=2,
                 size=300, mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentDetection, self).__init__()
        # image processors
        self.augment_image = AugmentImage((size, size), bkg_paths, mean)
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(pr.LoadImage(4), [0], [0]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


draw_boxes2D = DrawBoxData2D(['background', 'solar_panel'],
                             pr.ToImageBoxCoordinates())
if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from data_manager import CSVLoader
    from paz.models import SSD300
    import glob
    model = SSD300()
    prior_boxes = model.prior_boxes

    path = 'datasets/solar_panel/BoundingBox.txt'
    class_names = ['background', 'solar_panel']
    data_manager = CSVLoader(path, class_names)
    dataset = data_manager.load_data()
    home = os.path.expanduser('~')
    bkg_path = os.path.join(home, '.keras/paz/datasets/voc-backgrounds/')
    wild_card = bkg_path + '*.png'
    bkg_paths = glob.glob(wild_card)
    process = AugmentDetection(prior_boxes, bkg_paths)
    show_boxes2D = ShowBoxes(class_names, prior_boxes)
    for sample_arg in range(len(dataset)):
        sample = dataset[sample_arg]
        wrapped_outputs = process(sample)
        image = wrapped_outputs['inputs']['image']
        boxes = wrapped_outputs['labels']['boxes']
        image, boxes = show_boxes2D(image, boxes)
