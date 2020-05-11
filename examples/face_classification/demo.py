import argparse
from tensorflow.keras.models import load_model
from paz.models import HaarCascadeDetector
from paz.datasets import get_class_names
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from pipelines import FaceClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-d', '--detector_name', type=str,
                        default='frontalface_default')
    args = parser.parse_args()

    detector = HaarCascadeDetector(args.detector_name)
    labels = get_class_names('FER')
    classifier = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')
    offsets = (0, 0)

    pipeline = FaceClassifier(detector, classifier, labels, offsets)
    camera = Camera(args.camera_id)
    player = VideoPlayer((1280, 960), pipeline, camera)
    player.run()
