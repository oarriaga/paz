from ..abstract import SequentialProcessor, Processor
from .. import processors as pr
from ..backend.image import resize_image, BILINEAR
from ..backend.keypoints import normalize_keypoints2D


class PredictRGBMask(SequentialProcessor):
    """Predicts RGB mask from a segmentation model
    # Arguments
        model: Keras segmentation model.
        epsilon: Float. Values below this value would be replaced by 0.
    """
    def __init__(self, model, epsilon=0.15):
        super(PredictRGBMask, self).__init__()
        self.add(pr.ResizeImage(model.input_shape[1:3]))
        self.add(pr.NormalizeImage())
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(model))
        self.add(pr.Squeeze(0))
        self.add(pr.ReplaceLowerThanThreshold(epsilon))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))


class RGBMaskToObjectPoints3D(SequentialProcessor):
    """Predicts 3D keypoints from an RGB mask.
    # Arguments
        object_sizes: Array (3) determining the (width, height, depth)
    """
    def __init__(self, object_sizes):
        super(RGBMaskToObjectPoints3D, self).__init__()
        self.add(pr.GetNonZeroValues())
        self.add(pr.ImageToNormalizedDeviceCoordinates())
        self.add(pr.Scale(object_sizes / 2.0))


class RGBMaskToImagePoints2D(SequentialProcessor):
    """Predicts 2D image keypoints from an RGB mask.
    """
    def __init__(self):
        super(RGBMaskToImagePoints2D, self).__init__()
        self.add(pr.GetNonZeroArguments())
        self.add(pr.ArgumentsToImageKeypoints2D())


class Pix2Points(Processor):
    """Predicts RGB_mask and corresponding points2D and points3D.

    # Arguments
        model: Keras segmentation model.
        object_sizes: Array (3) determining the (width, height, depth)
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized to original shape.
        method: Interpolation method to use if resize is True.

    # Note
        Compare with and without RGB interpolation.
    """
    def __init__(self, model, object_sizes, epsilon=0.15,
                 resize=False, method=BILINEAR):
        self.model = model
        self.resize = resize
        self.method = method
        self.object_sizes = object_sizes
        self.predict_RGBMask = PredictRGBMask(model, epsilon)
        self.mask_to_points3D = RGBMaskToObjectPoints3D(self.object_sizes)
        self.mask_to_points2D = RGBMaskToImagePoints2D()
        self.wrap = pr.WrapOutput(['points2D', 'points3D', 'RGB_mask'])

    def call(self, image):
        RGB_mask = self.predict_RGBMask(image)
        if self.resize:
            H, W, num_channels = image.shape
            RGB_mask = resize_image(RGB_mask, (W, H), self.method)
        else:
            H, W = self.model.output_shape[1:3]
        points3D = self.mask_to_points3D(RGB_mask)
        points2D = self.mask_to_points2D(RGB_mask)
        points2D = normalize_keypoints2D(points2D, H, W)
        return self.wrap(points2D, points3D, RGB_mask)
