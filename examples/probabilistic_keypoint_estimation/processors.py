from paz.backend.image import draw_circle
from paz.backend.image.draw import GREEN
from paz.backend.image import resize_image
from paz import processors as pr
from paz.abstract import Processor
import numpy as np
from paz.backend.image import lincolor
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class PartitionKeypoints(Processor):
    """Partitions keypoints from shape ''[num_keypoints, 2]'' into a list of
        the form ''[(2, 1), (2, 1), ....]'' and length equal to the number of
        of_keypoints.
    """
    def __init__(self):
        super(PartitionKeypoints, self).__init__()

    def call(self, keypoints):
        keypoints = np.vsplit(keypoints, len(keypoints))
        keypoints = [np.squeeze(keypoint) for keypoint in keypoints]
        return (*keypoints, )


class ToNumpyArray(Processor):
    def __init__(self):
        super(ToNumpyArray, self).__init__()

    def call(self, predictions):
        return np.array(predictions)


class PredictDistributions(Processor):
    def __init__(self, model, preprocess=None):
        super(PredictDistributions, self).__init__()
        self.model = model
        self.preprocess = preprocess

    def call(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        distributions = self.model(x)
        return distributions


class ComputeMeans(Processor):
    def __init__(self):
        super(ComputeMeans, self).__init__()

    def call(self, distributions):
        keypoints = np.zeros((len(distributions), 2))
        for arg, distribution in enumerate(distributions):
            keypoints[arg] = distribution.mean()
        return keypoints


class ToProbabilityGrid(Processor):
    def __init__(self, grid):
        self.grid = grid

    def call(self, distribution):
        probability = distribution.prob(self.grid).numpy()[::-1, :]
        return probability


def build_figure():
    figure = Figure()
    canvas = FigureCanvas(figure)
    axis = figure.gca()
    axis.axis('off')
    figure.tight_layout(pad=0)
    axis.margins(0)
    # figure.canvas.draw()
    return figure, axis, canvas


def to_pixels(figure):
    figure.canvas.draw()
    image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return image


def interpolate_probability(probability, shape):
    normalization_constant = np.max(probability)
    probability = probability / normalization_constant
    probability = probability * 255.0
    probability = probability.astype('uint8')
    probability = resize_image(probability, shape)
    probability = probability / 255.0
    probability = probability * normalization_constant
    return probability


class DrawProbabilities(Processor):
    def __init__(self, num_keypoints, normalized=True):
        self.colors = lincolor(num_keypoints, normalized=normalized)
        self.figure, self.axis, self.canvas = build_figure()
        # self._figure, self._axis, self._canvas = build_figure()

    def call(self, image, probabilities):
        for color, probability in zip(self.colors, probabilities):
            cmap = sns.light_palette(color, input='hsl', as_cmap=True)
            probability = interpolate_probability(probability, image.shape[:2])
            self.axis.contour(probability, cmap=cmap, levels=np.arange(1, 50, 3))
        self.axis.imshow(image)
        contour = to_pixels(self.figure)
        # contour = resize_image(contour, (image.shape[:2]))
        # self._axis.imshow(image)
        # self._axis.imshow(contour)
        # new_image = to_pixels(self._figure)
        return contour


class PredictMeanDistribution(Processor):
    def __init__(self, model, preprocess=None):
        super(PredictMeanDistribution, self).__init__()
        print('Building graph...')
        self.num_keypoints = len(model.output_shape)
        # self.model = tf.function(model.mean)
        self.model = model
        self.preprocess = preprocess

    def call(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        distributions = self.model(x)
        keypoints = np.zeros((self.num_keypoints, 2))
        for arg, distribution in enumerate(distributions):
            keypoints[arg] = distribution.mean()
        return keypoints


def draw_circles(image, points, color=GREEN, radius=3):
    for point in points:
        draw_circle(image, point, color, radius)
    return image


if __name__ == '__main__':
    from facial_keypoints import FacialKeypoints
    from paz.backend.image import show_image
    from paz.abstract import SequentialProcessor

    data_manager = FacialKeypoints('dataset/', 'train')
    datasets = data_manager.load_data()
    augment_keypoints = SequentialProcessor()
    augment_keypoints.add(pr.RandomKeypointRotation())
    augment_keypoints.add(pr.RandomKeypointTranslation())
    for arg in range(100):
        original_image = datasets[0]['image'].copy()
        kp = datasets[0]['keypoints'].copy()
        original_image, kp = augment_keypoints(original_image, kp)
        original_image = draw_circles(original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
