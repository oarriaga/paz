from paz import processors as pr

from processors import PreprocessImage, Round, MasksToColors


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
