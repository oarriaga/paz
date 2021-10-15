from paz.abstract import Processor


class ImageToClosedOneBall(Processor):
    """Map image value from [0, 255] -> [-1, 1].
    """
    def __init__(self):
        super(ImageToClosedOneBall, self).__init__()

    def call(self, image):
        return (image / 127.5) - 1


class ClosedOneBallToImage(Processor):
    """Map normalized value from [-1, 1] -> [0, 255].
    """
    def __init__(self):
        super(ClosedOneBallToImage, self).__init__()

    def call(self, image):
        return (image + 1.0) * 127.5
