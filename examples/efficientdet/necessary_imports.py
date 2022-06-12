import numpy as np
from paz.abstract import Processor
from paz.backend.image.opencv_image import resize_image


# Taken from efficientdet -> /paz/paz/processors/image.py
B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3, 57.1, 58.4
BGR_IMAGENET_STDEV = (B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV)
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)


# Taken from efficientdet -> /paz/paz/processors/image.py
class DivideStandardDeviationImage(Processor):
    """Divide channel-wise standard deviation to image.
    # Arguments
        mean: List of length 3, containing the channel-wise mean.
    """
    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation
        super(DivideStandardDeviationImage, self).__init__()

    def call(self, image):
        return image / self.standard_deviation


# Taken from efficientdet -> /paz/paz/processors/image.py
class ScaledResize(Processor):
    """Resizes image by returning the scales to original image.
    # Arguments
        image_size: Int, desired size of the model input.
    # Returns
        output_images: Numpy array, image resized to match
        image size.
        image_scales: Numpy array, scale to reconstruct the
        raw image from the output_images.
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(ScaledResize, self).__init__()

    def call(self, image):
        """
        # Arguments
            image: Numpy array, raw input image.
        """
        crop_offset_y = np.array(0)
        crop_offset_x = np.array(0)
        height = np.array(image.shape[0]).astype('float32')
        width = np.array(image.shape[1]).astype('float32')
        image_scale_y = np.array(self.image_size).astype('float32') / height
        image_scale_x = np.array(self.image_size).astype('float32') / width
        image_scale = np.minimum(image_scale_x, image_scale_y)
        scaled_height = (height * image_scale).astype('int32')
        scaled_width = (width * image_scale).astype('int32')
        scaled_image = resize_image(image, (scaled_width, scaled_height))
        scaled_image = scaled_image[
                       crop_offset_y: crop_offset_y + self.image_size,
                       crop_offset_x: crop_offset_x + self.image_size,
                       :]
        output_images = np.zeros((self.image_size,
                                  self.image_size,
                                  image.shape[2]))
        output_images[:scaled_image.shape[0],
                      :scaled_image.shape[1],
                      :scaled_image.shape[2]] = scaled_image
        image_scale = 1 / image_scale
        output_images = output_images[np.newaxis]
        return output_images, image_scale


# Taken from efficientdet -> /paz/paz/processors/detection.py
class ScaleBox(Processor):
    """Scale box coordinates of the prediction.
    """
    def __init__(self, scales):
        super(ScaleBox, self).__init__()
        self.scales = scales

    def call(self, boxes):
        boxes = scale_box(boxes, self.scales)
        return boxes


# Taken from efficientdet -> /paz/paz/backend/boxes.py
def scale_box(predictions, image_scales=None):
    """
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """

    if image_scales is not None:
        boxes = predictions[:, :4]
        scales = image_scales[np.newaxis][np.newaxis]
        boxes = boxes * scales
        predictions = np.concatenate([boxes, predictions[:, 4:]], 1)
    return predictions


def incrementer(initial_value):
    """ Generates a counter variable
    # Yields:
        counter_var: Int a counter vaiable
    """
    counter_var = initial_value
    while True:
        yield counter_var
        counter_var += 1
