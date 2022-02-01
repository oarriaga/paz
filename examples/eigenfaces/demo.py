import os
import argparse
import numpy as np
import processors as pe
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from pipelines import DetectEigenFaces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    parser.add_argument('-e', '--experiments_path', type=str,
                        default='experiments',
                        help='Directory for writing and loading experiments')
    parser.add_argument('-d', '--database_path', type=str,
                        default='database',
                        help='Directory for the database')
    args = parser.parse_args()

    if not os.path.exists(args.experiments_path):
        os.makedirs(args.experiments_path)
    if not os.path.exists(args.database_path):
        os.makedirs(args.database_path)

    #  check if eigenfaces and mean face are already computed
    needed_files = ['eigenvalues.npy', 'eigenfaces.npy', 'mean_face.npy']
    if set(os.listdir(args.experiments_path)) != set(needed_files):
        raise FileNotFoundError('''Need necessary files to run the demo. Please
                                run eigenface.py first and then try running the
                                demo.''')

    #  check if database is available
    needed_files = ['images', 'database.npy']
    if set(os.listdir(args.database_path)) != set(needed_files):
        raise FileNotFoundError('''Need database to run the demo. Please
                                update the database with database.py first
                                and then try running the demo.''')

    eigenfaces = np.load(os.path.join(args.experiments_path, 'eigenfaces.npy'))
    mean_face = np.load(os.path.join(args.experiments_path, 'mean_face.npy'))

    database_path = os.path.join(args.database_path, 'database.npy')
    weights = np.load(database_path, allow_pickle=True).item()

    # user defined parameters
    thresh = 1e4
    norm_order = 2
    # measure = pe.CalculateNorm(norm_order)
    measure = pe.CalculateCosineSimilarity()

    pipeline = DetectEigenFaces(weights, measure, thresh, eigenfaces,
                                mean_face, [args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
