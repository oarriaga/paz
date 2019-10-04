from ...core import ops


class HaarCascadeDetector(object):
    """Haar cascade face detector.
    # Arguments
        path: String. Path to default openCV XML format.
        scale_factor= Float. Scale for image reduction
        min_neighbors: Int.
    # Methods
        predict()
    """
    def __init__(self, path, scale_factor=1.3, min_neighbors=5):
        self.model = ops.cascade_classifier(path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def predict(self, gray_image):
        """ Detects faces from gray images.
        """
        if len(gray_image.shape) != 2:
            raise ValueError('Invalid gray image shape:', gray_image.shape)
        args = (gray_image, self.scale_factor, self.min_neighbors)
        return self.model.detectMultiScale(*args)
