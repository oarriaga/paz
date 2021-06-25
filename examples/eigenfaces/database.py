import os
import argparse
import numpy as np
import processors as pe
from paz.abstract import Loader
from paz.backend.image import load_image
from pipelines import CalculateFaceWeights
from demo_pipeline import CalculateTestFaceWeights


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
    def __init__(self, path, label, project, mean_face):
        self.path = path
        self.label = label
        self.data = LoadTestData(self.path, self.label)
        self.calculate_weights = CalculateTestFaceWeights(project, mean_face)

    def add_to_database(self):
        data = self.data.load_data()
        for sample in data:
            image, label = sample['image'], sample['label']
            weight = self.calculate_weights(image)
            weight = np.array(weight[np.newaxis].T)
            updated_database = update_dictionary(database, label, weight)
        return updated_database


def update_dictionary(dictionary, key, values):
    if key not in dictionary:
        dictionary[key] = values
    else:
        dictionary[key] = np.hstack((dictionary[key], values))
    return dictionary


if __name__ == "__main__":
    description = 'Update database for eigen face classification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e', '--experiments_path', type=str,
                        default='experiments',
                        help='Directory for writing and loading experiments')
    parser.add_argument('-d', '--database_path', type=str,
                        default='image_database',
                        help='Directory for all the images')
    args = parser.parse_args()

    eigenfaces = np.load(os.path.join(args.experiments_path, 'eigenfaces.npy'))
    mean_face = np.load(os.path.join(args.experiments_path, 'mean_face.npy'))
    project = CalculateFaceWeights(eigenfaces, mean_face, with_crop=False)

    database = np.load(os.path.join(args.experiments_path, 'database.npy'),
                       allow_pickle=True).item()

    # provide image path here
    Octavio = '/home/proneet/DFKI/paz/examples/eigenfaces/test_image/octavio'
    Proneet = '/home/proneet/DFKI/paz/examples/eigenfaces/test_image/proneet'

    data_path = {'Octavio': os.path.join(args.database_path, 'octavio'),
                 'Proneet': os.path.join(args.database_path, 'proneet')
                }

    database_path = os.path.join(args.experiments_path, 'database.npy')
    for each in data_path:
        database_object = Database(data_path[each], each, project, mean_face)
        weights_database = database_object.add_to_database()
        # np.save(database_path, weights_database)
