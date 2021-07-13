import numpy as np
from paz import processors as pr
from paz.backend.image import lincolor
from paz.pipelines import HaarCascadeFrontalFace
from paz.abstract import SequentialProcessor, Processor
from pipelines import CalculateFaceWeights
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
