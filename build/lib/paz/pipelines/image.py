from ..abstract import SequentialProcessor
from .. import processors as pr


class AugmentImage(SequentialProcessor):
    """Augments an RGB image by randomly changing contrast, brightness
        saturation and hue.
    """
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation())
        self.add(pr.RandomHue())


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


class AutoEncoderPredictor(SequentialProcessor):
    """Pipeline for predicting values from an auto-encoder.

    # Arguments
        model: Keras model.
    """
    def __init__(self, model):
        super(AutoEncoderPredictor, self).__init__()
        preprocess = SequentialProcessor(
            [pr.ResizeImage(model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.NormalizeImage(),
             pr.ExpandDims(0)])
        self.add(pr.Predict(model, preprocess))
        self.add(pr.Squeeze(0))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))
        self.add(pr.WrapOutput(['image']))


class EncoderPredictor(SequentialProcessor):
    """Pipeline for predicting latent vector of an encoder.

    # Arguments
        model: Keras model.
    """
    def __init__(self, encoder):
        super(EncoderPredictor, self).__init__()
        self.encoder = encoder
        preprocess = SequentialProcessor([
            pr.ConvertColorSpace(pr.RGB2BGR),
            pr.ResizeImage(encoder.input_shape[1:3]),
            pr.NormalizeImage(),
            pr.ExpandDims(0)])
        self.add(pr.Predict(encoder, preprocess, pr.Squeeze(0)))


class DecoderPredictor(SequentialProcessor):
    """Pipeline for predicting decoded image from a latent vector.

    # Arguments
        model: Keras model.
    """
    def __init__(self, decoder):
        self.decoder = decoder
        super(DecoderPredictor, self).__init__()
        self.add(pr.Predict(decoder, pr.ExpandDims(0), pr.Squeeze(0)))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))
        self.add(pr.ConvertColorSpace(pr.BGR2RGB))
