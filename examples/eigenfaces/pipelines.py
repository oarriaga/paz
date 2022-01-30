import numpy as np
import processors as pe
from paz import processors as pr
from paz.backend.image import lincolor
from paz.pipelines import HaarCascadeFrontalFace
from paz.abstract import SequentialProcessor, Processor
from processors import MeasureSimilarity


class DetectEigenFaces(Processor):
    def __init__(self, weights, measure, thresh, eigenfaces,
                 mean_face, offsets=[0, 0]):
        super(DetectEigenFaces, self).__init__()
        self.offsets = offsets
        self.class_names = list(weights.keys()) + ['Face not found']
        self.colors = lincolor(len(self.class_names))
        self.croped_images = None
        # detection
        self.detect = HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.face_detector = EigenFaceDetector(weights, measure, thresh,
                                               eigenfaces, mean_face)
        # drawing and wrapping
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors,
                                   weighted=True, with_score=False)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image.copy())['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        self.cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(self.cropped_images, boxes2D):
            box2D.class_name = self.face_detector(cropped_image)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class EigenFaceDetector(Processor):
    def __init__(self, weights_data_base, measure, thresh,
                 eigenfaces, mean_face):
        self.weights_data_base = weights_data_base
        self.calculate_weights = CalculateFaceWeights(eigenfaces, mean_face,
                                                      with_crop=False)
        self.query = QueryFace(measure, thresh)
        super(EigenFaceDetector, self).__init__()

    def call(self, image):
        test_weight = self.calculate_weights(image)
        similar_face = self.query(test_weight, self.weights_data_base)
        return similar_face


class QueryFace(Processor):
    """Identify the most similar face in the database

    # Properties
        measure: Similarity measurement metric
        thresh: Float. Threshold for the similarity between two images
    """

    def __init__(self, measure, thresh):
        self.thresh = thresh
        self.measure_distance = MeasureSimilarity(measure)
        super(QueryFace).__init__()

    def call(self, test_face_weight, database):
        self.database = database
        weights_difference = []
        for sample in database:
            weight = database[sample].T
            weight_norm = self.measure_distance(weight, test_face_weight)
            weights_difference.append(np.min(weight_norm))

        if np.min(weights_difference) > self.thresh:
            return 'Face not found'
        else:
            most_similar_face_arg = np.argmin(weights_difference)
        return list(database.keys())[most_similar_face_arg]


class CalculateEigenFaces(pr.SequentialProcessor):
    def __init__(self, total_variance=0.95):
        super(CalculateEigenFaces, self).__init__()
        self.total_variance = total_variance
        if not (0 < self.total_variance <= 1.0):
            raise ValueError('Variance must be in (0, 1]')
        self.add(pr.ControlMap(pe.CalculateMeanFace(), [0], [1], {0: 0}))
        self.add(pr.ControlMap(pe.SubtractMeanFace(), [0, 1], [0], {1: 1}))
        self.add(pr.ControlMap(pe.ReshapeFacesToVectors()))
        self.add(pr.ControlMap(pe.ComputeCovarianceMatrix()))
        self.add(pr.ControlMap(pe.ComputeEigenvectors(), [0], [0, 1]))
        self.add(pr.ControlMap(pe.ToDescendingOrder(), [0, 1], [0, 1]))
        filter_variance = pe.FilterVariance(self.total_variance)
        self.add(pr.ControlMap(filter_variance, [0, 1], [0, 1]))


class PostrocessEigenFace(pr.SequentialProcessor):
    def __init__(self, shape=(48, 48)):
        self.shape = shape
        super(PostrocessEigenFace, self).__init__()
        self.add(pe.MinMaxNormalization(255.0))
        self.add(pe.Reshape(shape))
        self.add(pr.CastImage('uint8'))


class ProjectVectorToBase(pr.SequentialProcessor):
    def __init__(self, base, mean_face):
        super(ProjectVectorToBase, self).__init__()
        self.base, self.mean_face = base, mean_face
        self.add(pr.ControlMap(pr.ExpandDims(-1)))
        self.add(pe.SubtractMeanFace())
        self.add(pe.ComputeWeights(self.base))


class CalculateFaceWeights(pr.Processor):
    def __init__(self, base, mean_face, shape=(48, 48), with_crop=True):
        super(CalculateFaceWeights, self).__init__()
        self.base, self.mean_face = base, mean_face
        self.preprocess = pr.SequentialProcessor()
        self.convert_to_gray = pr.ConvertColorSpace(pr.RGB2GRAY)
        if with_crop:
            self.preprocess.add(pe.CropFrontalFace())
        self.preprocess.add(pr.ResizeImage(shape))
        self.preprocess.add(pr.ExpandDims(-1))
        self.subtract = pe.SubtractMeanFace()
        self.project = pe.ProjectToBase(self.base)

    def call(self, face):
        if len(face.shape) != 3:
            raise ValueError('input should have shape [H, W, num_channels]')
        if face.shape[-1] == 3:
            face = self.convert_to_gray(face)
        face = self.preprocess(face)
        face = self.subtract(face, self.mean_face)
        weights = self.project(face)
        return weights
