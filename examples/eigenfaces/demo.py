import os
import argparse
import numpy as np
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from demo_pipeline import DetectEigenFaces


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

    database_path = os.path.join(args.experiments_path, 'database.npy')
    weights = np.load(database_path, allow_pickle=True).item()

    # user defined parameters
    norm_order = 2
    thresh = 1e6

    pipeline = DetectEigenFaces(weights, norm_order, thresh, eigenfaces,
                                mean_face, [args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
