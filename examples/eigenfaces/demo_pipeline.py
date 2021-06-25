import os
import argparse
import numpy as np
import processors as pe
from paz.abstract import Loader
from paz import processors as pr
from paz.pipelines import HaarCascadeFrontalFace
from paz.abstract import SequentialProcessor, Processor
from paz.backend.image import load_image

description = 'Eigenfaces demo pipeline'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-e', '--experiments_path', type=str,
                    default='experiments',
                    help='Directory for writing and loading experiments')
args = parser.parse_args()

database = np.load(os.path.join(args.experiments_path, 'database.npy'),
                   allow_pickle=True).item()


class DetectEigenFaces(Processor):
    def __init__(self, weights_data_base, parameters, offsets=[0, 0],
                 colors=[[255, 0, 0], [0, 255, 0]]):
        super(DetectEigenFaces, self).__init__()
        self.offsets = offsets
        self.colors = colors
        # self.class_names = class_names
        self.croped_images = None
        # detection
        self.detect = HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.class_names = parameters['class_names']
        self.face_detector = EigenFaceDetector(weights_data_base, parameters)

        # drawing and wrapping
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image.copy())['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        self.cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(self.cropped_images, boxes2D):
            self.face_detector.store_each_frame(cropped_image)
            self.face_detector()
            box2D.class_name = self.face_detector()
            # box2D.score = np.amax(predictions['scores'])
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class EigenFaceDetector(Processor):
    def __init__(self, weights_data_base, parameters):
        self.weights_data_base = weights_data_base
        self.calculate_weights = CalculateFaceWeights(parameters)
        self.query = QueryFace(parameters)
        super(EigenFaceDetector, self).__init__()

    def store_each_frame(self, test_data):
        self.test_data = test_data

    def call(self):
        test_weight = self.calculate_weights(self.test_data)
        similar_face = self.query(test_weight, self.weights_data_base)
        return similar_face


class LoadTestData(Loader):
    def __init__(self, path, label, image_size=(48, 48)):
        self.images_path = path
        self.image_size = image_size
        self.label = label
        self.crop = pe.CropFrontalFace()
        super(LoadTestData, self).__init__(path, label, None, None)

    def load_data(self):
        data = []
        for filename in os.listdir(self.images_path):
            face = load_image(os.path.join(self.images_path, filename))
            face = self.crop(face)
            sample = {'image': face, 'label': self.label}
            data.append(sample)
        return data


class Database():
    def __init__(self, path, label, parameters):
        self.path = path
        self.label = label
        self.data = LoadTestData(self.path, self.label)
        self.calculate_weights = CalculateFaceWeights(parameters)

    def add_to_database(self):
        data = self.data.load_data()
        new_database = []
        for sample in data:
            image, label = sample['image'], sample['label']
            weight = self.calculate_weights(image)
            # weight = np.array(weight[np.newaxis].T)
            self.new_data = {'label': label, 'weight': weight}
            # self.new_data_array = {label: np.array(weight)}
            new_database.append(self.new_data)
            # u_database = update_dictionary(database, label, weight)
        # np.save(os.path.join(args.experiments_path, 'database.npy'), u_database)
        return new_database


def update_dictionary(dictionary, key, values):
    if key not in dictionary:
        dictionary[key] = values
    else:
        dictionary[key] = np.hstack((dictionary[key], values))
    return dictionary


class QueryFace(Processor):
    def __init__(self, parameters):
        self.norm = np.linalg.norm
        self.norm_order = parameters['norm_order']
        self.threshold = parameters['threshold']
        super(QueryFace).__init__()

    def call(self, test_face_weight, database):
        # you could also get a none
        self.database = database
        weights_difference = []
        for sample in database:
            weight = sample['weight']
            weight_norm = self.norm((weight - test_face_weight),
                                     ord=self.norm_order)
            weights_difference.append(weight_norm)

        if np.min(weights_difference) < self.threshold:
            return None
        else:
            most_similar_face_arg = np.argmin(weights_difference)
        return database[most_similar_face_arg]["label"]


class CalculateFaceWeights(pr.Processor):
    def __init__(self, parameters):
        super(CalculateFaceWeights, self).__init__()
        self.project = parameters['project']
        self.norm = np.linalg.norm
        self.norm_order = parameters['norm_order']
        self.mean_face = parameters['mean_face']
        self.image_shape = parameters['image_shape']
        self.convert_to_gray = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.preprocess = pr.SequentialProcessor()
        self.preprocess.add(pr.ResizeImage(self.image_shape))
        self.preprocess.add(pr.ExpandDims(-1))
        self.subtract = pe.SubtractMeanFace()

    def call(self, face):
        if len(face.shape) != 3:
            raise ValueError('input should have shape [H, W, num_channels]')
        if face.shape[-1] == 3:
            face = self.convert_to_gray(face)
        face = self.preprocess(face)
        face = self.subtract(face, self.mean_face)
        weights = self.project(face)
        return weights


