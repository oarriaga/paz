import numpy as np
from matplotlib import offsetbox
import matplotlib.pyplot as plt
from paz import processors as pr
from paz.pipelines import HaarCascadeFrontalFace


class ExtractFaces(pr.Processor):
    def __init__(self, shape=(48, 48)):
        super(ExtractFaces, self).__init__()
        self.shape = shape

    def call(self, data):
        num_samples = len(data)
        faces = np.zeros((self.shape[0], self.shape[1], num_samples))
        for sample_arg in range(len(data)):
            faces[..., sample_arg] = data[sample_arg]['image']
        return faces


class CalculateMeanFace(pr.Processor):
    def __init__(self):
        super(CalculateMeanFace, self).__init__()

    def call(self, faces):
        return np.mean(faces, axis=-1)


class SubtractMeanFace(pr.Processor):
    def __init__(self):
        super(SubtractMeanFace, self).__init__()

    def call(self, faces, mean_face):
        faces = faces - np.expand_dims(mean_face, -1)
        return faces


class ReshapeFacesToVectors(pr.Processor):
    def __init__(self):
        super(ReshapeFacesToVectors, self).__init__()

    def call(self, faces):
        num_samples = faces.shape[-1]
        faces = faces.reshape(-1, num_samples)
        return faces


class ComputeCovarianceMatrix(pr.Processor):
    def __init__(self):
        super(ComputeCovarianceMatrix, self).__init__()

    def call(self, faces):
        covariance_matrix = np.cov(faces)
        return covariance_matrix


class ComputeEigenvectors(pr.Processor):
    def __init__(self):
        super(ComputeEigenvectors, self).__init__()

    def call(self, covariance_matrix):
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        return eigen_values, eigen_vectors.T


class ToDescendingOrder(pr.Processor):
    def __init__(self):
        super(ToDescendingOrder, self).__init__()

    def call(self, eigenvalues, eigenvectors):
        return eigenvalues[::-1], eigenvectors[::-1]


class MinMaxNormalization(pr.Processor):
    def __init__(self, scale=1.0):
        super(MinMaxNormalization, self).__init__()
        self.scale = scale

    def call(self, vector):
        minimum, maximum = vector.min(), vector.max()
        return self.scale * ((vector - minimum) / (maximum - minimum))


class Reshape(pr.Processor):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def call(self, vector):
        return np.reshape(vector, self.shape)


class ProjectToBase(pr.Processor):
    def __init__(self, base):
        super(ProjectToBase, self).__init__()
        self.base = base

    def call(self, face):
        face = np.expand_dims(face, -1)
        face = face.reshape(-1, 1)
        weights = np.matmul(self.base, face)
        weights = np.squeeze(weights)
        return weights


class FilterVariance(pr.Processor):
    def __init__(self, total_variance):
        super(FilterVariance, self).__init__()
        self.total_variance = total_variance
        if not (0 < self.total_variance <= 1.0):
            raise ValueError('Variance must be in (0, 1]')

    def call(self, eigenvalues, eigenvectors):
        normalized_eigenvalues = eigenvalues / eigenvalues.sum()
        cumsum_variances = np.cumsum(normalized_eigenvalues)
        mask = cumsum_variances < self.total_variance
        return eigenvalues[mask], eigenvectors[mask]


class PlotEmbeddings(pr.Processor):
    def __init__(self, title=None, epsilon=4e-3, cmap=plt.cm.gray):
        super(PlotEmbeddings, self).__init__()
        self.title = title
        self.epsilon = epsilon
        self.cmap = cmap

    def call(self, x, images):
        x_min, x_max = np.min(x, 0), np.max(x, 0)
        x = (x - x_min) / (x_max - x_min)
        plt.figure()
        axis = plt.subplot(111)
        if hasattr(offsetbox, 'AnnotationBbox'):
            num_samples = len(x)
            shown_images = np.array([[1.0, 1.0]])
            for sample_arg in range(num_samples):
                sample, image = x[sample_arg], images[sample_arg]
                distance = (sample - shown_images)**2
                distance = np.sum(distance, 1)
                if np.min(distance) < self.epsilon:
                    continue
                shown_images = np.r_[shown_images, [sample]]
                box = offsetbox.OffsetImage(image, cmap=self.cmap)
                imagebox = offsetbox.AnnotationBbox(box, sample)
                axis.add_artist(imagebox)
        plt.xticks([])
        plt.yticks([])
        if self.title is not None:
            plt.title(self.title)


class CropFrontalFace(pr.Processor):
    def __init__(self):
        super(CropFrontalFace, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        if len(boxes2D) == 1:
            image = self.crop(image, boxes2D)[0]
        return image


class MeasureSimilarity(pr.Processor):
    def __init__(self, measure):
        super(MeasureSimilarity, self).__init__()
        self.measure = measure

    def call(self, vector_1, vector_2):
        similarity = self.measure(vector_1, vector_2)
        return similarity


class CalculateNorm(pr.Processor):
    def __init__(self, norm_order):
        super(CalculateNorm, self).__init__()
        self.order = norm_order
        self.norm = np.linalg.norm

    def call(self, vector_1, vector_2):
        distance = self.norm((vector_1 - vector_2), ord=self.order, axis=1)
        return distance


class CalculateCosineSimilarity(pr.Processor):
    def __init__(self):
        super(CalculateCosineSimilarity, self).__init__()
        self.norm = np.linalg.norm

    def call(self, vector_1, vector_2):
        distance = np.dot(vector_1, vector_2) / (self.norm(vector_1) *
                                                 self.norm(vector_2))
        return distance
