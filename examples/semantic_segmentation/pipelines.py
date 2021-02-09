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
