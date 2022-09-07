import argparse
from paz.abstract import SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
from paz.applications import SSD512MinimalHandPose
from paz import processors as pr


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)


class HandStateEstimation(SequentialProcessor):
    def __init__(self, camera):
        super(HandStateEstimation, self).__init__()
        intro_topics = ['image', 'boxes2D', 'keypoints2D', 'keypoints3D']
        self.add(SSD512MinimalHandPose())
        self.add(pr.UnpackDictionary(intro_topics))
        self.add(pr.ControlMap(
            pr.Translation3DFromBoxWidth(camera), [1], [4], {1: 1}))
        outro_topics = intro_topics + ['translation3D']
        self.add(pr.WrapOutput(outro_topics))


pipeline = HandStateEstimation(camera)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
