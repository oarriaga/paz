import os
import argparse
import numpy as np
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from demo_pipeline import DetectEigenFaces
from pipelines import CalculateFaceWeights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    parser.add_argument('-e', '--experiments_path', type=str,
                        default='experiments',
                        help='Directory for writing and loading experiments')
    args = parser.parse_args()

    eigenfaces = np.load(os.path.join(args.experiments_path, 'eigenfaces.npy'))
    mean_face = np.load(os.path.join(args.experiments_path, 'mean_face.npy'))
    project = CalculateFaceWeights(eigenfaces, mean_face, with_crop=False)

    database_path = os.path.join(args.experiments_path, 'database.npy')
    weights_database = np.load(database_path, allow_pickle=True).item()

    class_names = list(weights_database.keys())
    colors = [[255, 0, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]]

    # user defined parameters
    parameters = {'norm_order': 2,
                  'threshold': 1e-3,
                  'class_names': class_names,
                  'colors': colors
                 }

    pipeline = DetectEigenFaces(weights_database, parameters, project,
                                mean_face, [args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
