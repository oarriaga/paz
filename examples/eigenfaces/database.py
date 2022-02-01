import os
import argparse
import numpy as np
import processors as pe
from paz import processors as pr
from paz.backend.image import load_image
from pipelines import CalculateFaceWeights


class Database():
    """Load images and updates the database with their weights value

    # Properties
        path: String. Path of the image directory
        label: String. label for the images

    # Methods
        load_data()
        update_dictionary()
        add_to_database()
    """

    def __init__(self, path, label, eigenfaces, mean_face):
        self.path = path
        self.label = label
        self.crop = pe.CropFrontalFace()
        self.expand_dimension = pr.ExpandDims(-1)
        self.project = CalculateFaceWeights(eigenfaces, mean_face,
                                            with_crop=False)
        super(Database, self).__init__()

    def load_data(self, path, label, with_crop=True):
        data = []
        for filename in os.listdir(path):
            face = load_image(os.path.join(path, filename))
            if with_crop:
                croped_face = self.crop(face)
            sample = {'image': croped_face, 'label': label}
            data.append(sample)
        return data

    def update_dictionary(self, dictionary, key, values):
        if key not in dictionary:
            dictionary[key] = values
        else:
            dictionary[key] = np.hstack((dictionary[key], values))
        return dictionary

    def add_to_database(self, database=None, is_new_database=False):
        data = self.load_data(self.path, self.label)
        for sample in data:
            image, label = sample['image'], sample['label']
            weight = self.project(image)
            weight = self.expand_dimension(weight)
            if is_new_database:
                database = {label: weight}
                is_new_database = False
            else:
                database = self.update_dictionary(database, label, weight)
        return database


if __name__ == "__main__":
    description = 'Update database for eigenface classification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e', '--experiments_path', type=str,
                        default='experiments',
                        help='Directory for writing and loading experiments')
    parser.add_argument('-d', '--database_path', type=str,
                        default='database',
                        help='Directory for the database')
    args = parser.parse_args()

    image_path = os.path.join(args.database_path, 'images')
    database_path = os.path.join(args.database_path, 'database.npy')

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    needed_files = ['eigenvalues.npy', 'eigenfaces.npy', 'mean_face.npy']
    if set(os.listdir(args.experiments_path)) != set(needed_files):
        raise FileNotFoundError('''Need necessary files to create database.
                                Please run eigenface.py first''')

    eigenfaces = np.load(os.path.join(args.experiments_path, 'eigenfaces.npy'))
    mean_face = np.load(os.path.join(args.experiments_path, 'mean_face.npy'))

    labels = os.listdir(os.path.join(args.database_path, 'images'))
    if len(labels) == 0:
        raise FileNotFoundError('''No image available to create database.
        To add images follow the directory structure provided in README.md''')

    database_files = 'database.npy'

    write_to_file = True
    for label in labels:
        data_path = os.path.join(args.database_path, 'images', label)
        update_database = Database(data_path, label, eigenfaces, mean_face)
        if database_files not in os.listdir(args.database_path):
            weights_database = update_database.add_to_database(
                is_new_database=True)
        else:
            database = np.load(database_path, allow_pickle=True).item()
            weights_database = update_database.add_to_database(database)
        if write_to_file:
            np.save(database_path, weights_database)
