from ..abstract import SequentialProcessor
from .. import processors as pr
from ..backend.image import get_affine_transform


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


class PreprocessImageHigherHRNet(pr.Processor):
    """Transform the image according to the HigherHRNet model requirement.
    # Arguments
        scaling_factor: Int. scale factor for image dimensions.
        input_size: Int. resize the first dimension of image to input size.
        inverse: Boolean. Reverse the affine transform input.
        image: Numpy array. Input image

    # Returns
        image: resized and transformed image
        center: center of the image
        scale: scaled image dimensions
    """
    def __init__(self, scaling_factor=200, input_size=512, multiple=64):
        super(PreprocessImageHigherHRNet, self).__init__()
        self.get_image_center = pr.GetImageCenter()
        self.get_size = pr.GetTransformationSize(input_size, multiple)
        self.get_scale = pr.GetTransformationScale(scaling_factor)
        self.get_source_destination_point = pr.GetSourceDestinationPoints(
            scaling_factor)
        self.transform_image = pr.SequentialProcessor(
            [pr.WarpAffine(), pr.ImagenetPreprocessInput(), pr.ExpandDims(0)])

    def call(self, image):
        center = self.get_image_center(image)
        size = self.get_size(image)
        scale = self.get_scale(image, size)
        source_point, destination_point = self.get_source_destination_point(
            center, scale, size)
        transform = get_affine_transform(source_point, destination_point)
        image = self.transform_image(image, transform, size)
        return image, center, scale
