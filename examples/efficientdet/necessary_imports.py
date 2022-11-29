import numpy as np
from paz.abstract import Processor
from paz.backend.image.opencv_image import resize_image

B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3, 57.1, 58.4
BGR_IMAGENET_STDEV = (B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV)
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)


class DivideStandardDeviationImage(Processor):
    """Divides image by channel-wise standard deviation.

    # Arguments
        standard_deviation: List, channel-wise standard deviation.
    """
    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation
        super(DivideStandardDeviationImage, self).__init__()

    def call(self, image):
        return image / self.standard_deviation


class ScaledResize(Processor):
    """Resizes image.

    # Arguments
        image_size: Int, desired model's input size.

    # Returns
        output_images: Array, resized image.
        image_scales: Array, scales to reconstruct raw image.
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(ScaledResize, self).__init__()

    def compute_image_scale(self, image):
        """Computes image resizing scale.

        # Arguments
            image: Array, raw input image.

        # Returns
            Tuple: holding width, height and image_scale.
        """
        height = np.array(image.shape[0]).astype('float32')
        width = np.array(image.shape[1]).astype('float32')
        image_scale_y = np.array(self.image_size).astype('float32') / height
        image_scale_x = np.array(self.image_size).astype('float32') / width
        image_scale = np.minimum(image_scale_x, image_scale_y)
        return width, height, image_scale

    def scale_image(self, image, width, height, image_scale):
        """Scales image using computed scale.

        # Arguments
            image: Array, raw input image.
            width: Array, raw image width.
            height: Array, raw image height.
            image_scale: Array, scale to resize raw image.

        # Returns
            scaled_image: Array, scaled input image.
        """
        scaled_height = (height * image_scale).astype('int32')
        scaled_width = (width * image_scale).astype('int32')
        scaled_image = resize_image(image, (scaled_width, scaled_height))
        return scaled_image

    def crop_image(self, scaled_image, crop_offset_x, crop_offset_y):
        """Crops given image.

        # Arguments
            scaled_image: Array, input image.
            crop_offset_x: Array, x crop offset.
            crop_offset_y: Array, y crop offset.

        # Returns
            cropped_image: Array, cropped input image.
        """
        cropped_image = scaled_image[
                        crop_offset_y: crop_offset_y + self.image_size,
                        crop_offset_x: crop_offset_x + self.image_size,
                        :]
        return cropped_image

    def compose_output(self, image, scaled_image, image_scale):
        """Composes output image and image scale.

        # Arguments
            image: Array, raw input image.
            scaled_image: Array, scaled input image.
            image_scale: Array, scale to resize raw image.

        # Returns
            Tuple: holding output images and image scale.
        """
        output_images = np.zeros(
            (self.image_size, self.image_size, image.shape[2]))
        output_images[:scaled_image.shape[0],
                      :scaled_image.shape[1],
                      :scaled_image.shape[2]] = scaled_image
        image_scale = 1 / image_scale
        output_images = output_images[np.newaxis]
        return output_images, image_scale

    def call(self, image):
        """
        # Arguments
            image: Array, raw input image.

        # Returns:
            Tuple: holding the output image and image scale.
        """
        width, height, image_scale = self.compute_image_scale(image)
        scaled_image = self.scale_image(image, width, height, image_scale)
        crop_offset_x, crop_offset_y = np.array(0), np.array(0)
        scaled_image = self.crop_image(
            scaled_image, crop_offset_x, crop_offset_y)
        output_images, image_scale = self.compose_output(
            self, image, scaled_image, image_scale)
        return output_images, image_scale


class ScaleBox(Processor):
    """Scales prediction box coordinates.
    """
    def __init__(self, scales):
        super(ScaleBox, self).__init__()
        self.scales = scales

    def call(self, boxes):
        boxes = scale_box(boxes, self.scales)
        return boxes


def scale_box(predictions, image_scales=None):
    """Scales boxes according to image_scales.

    # Arguments
        predictions: Array, prediction boxes.
        image_scales: Array, scales to reconstruct raw image.

    # Returns
        Array of shape `[num_boxes, N]`.
    """

    if image_scales is not None:
        boxes = predictions[:, :4]
        scales = image_scales[np.newaxis][np.newaxis]
        boxes = boxes * scales
        predictions = np.concatenate([boxes, predictions[:, 4:]], 1)
    return predictions
