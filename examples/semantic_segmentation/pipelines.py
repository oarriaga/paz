from paz import processors as pr

from processors import PreprocessImage, Round, MasksToColors
from processors import FromIdToMask, ResizeImageWithNearestNeighbors


class PostprocessSegmentation(pr.SequentialProcessor):
    def __init__(self, model, colors=None):
        super(PostprocessSegmentation, self).__init__()
        self.add(PreprocessImage())
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(model))
        self.add(pr.Squeeze(0))
        self.add(Round())
        self.add(MasksToColors(model.output_shape[-1], colors))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))
        self.add(pr.ShowImage())


class PreprocessSegmentation(pr.SequentialProcessor):
    def __init__(self, image_shape, num_classes, input_name='input_1'):
        super(PreprocessSegmentation, self).__init__()
        H, W = image_shape
        preprocess_image = PreprocessImage()
        self.add(pr.UnpackDictionary(['image', 'masks']))
        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.SequenceWrapper({0: {input_name: [H, W, 3]}},
                                    {1: {'masks': [H, W, num_classes]}}))


class PreprocessSegmentationIds(pr.SequentialProcessor):
    def __init__(self, image_shape, num_classes, input_name='input_1'):
        super(PreprocessSegmentationIds, self).__init__()
        self.add(pr.UnpackDictionary(['image_path', 'label_path']))
        preprocess_image = pr.SequentialProcessor()
        preprocess_image.add(pr.LoadImage())
        preprocess_image.add(pr.ResizeImage(image_shape))
        preprocess_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        preprocess_image.add(pr.SubtractMeanImage(pr.BGR_IMAGENET_MEAN))

        preprocess_label = pr.SequentialProcessor()
        preprocess_label.add(pr.LoadImage())
        preprocess_label.add(ResizeImageWithNearestNeighbors(image_shape))
        preprocess_label.add(FromIdToMask())

        self.add(pr.ControlMap(preprocess_image, [0], [0]))
        self.add(pr.ControlMap(preprocess_label, [1], [1]))
        H, W = image_shape[:2]
        self.add(pr.SequenceWrapper({0: {input_name: [H, W, 3]}},
                                    {1: {'masks': [H, W, num_classes]}}))


class PostprocessSegmentationIds(pr.SequentialProcessor):
    def __init__(self, num_classes, colors=None):
        super(PostprocessSegmentationIds, self).__init__()
        self.add(MasksToColors(num_classes, colors))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))


class PostProcessImage(pr.SequentialProcessor):
    def __init__(self):
        super(PostProcessImage, self).__init__()
        self.add(pr.AddMeanImage(pr.BGR_IMAGENET_MEAN))
        self.add(pr.CastImage('uint8'))
        self.add(pr.ConvertColorSpace(pr.BGR2RGB))
