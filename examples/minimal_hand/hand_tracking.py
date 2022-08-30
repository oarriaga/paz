import argparse
from paz.applications import DetectMinimalHand
from paz.applications import MinimalHandPoseEstimation
from paz.pipelines.detection import SSD512HandDetection
from paz.backend.camera import VideoPlayer, Camera
from paz.abstract import SequentialProcessor
from paz import processors as pr


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=70, help='Horizontal field of view in degrees')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
focal_length = camera.intrinsics[0, 0]


class EstimateHandPosition(pr.Processor):
    def __init__(self, focal_length, hand_width=10):
        super(EstimateHandPosition, self).__init__()
        self.focal_length = focal_length
        self.hand_width = hand_width

    def call(self, boxes2D):
        hands_center = []
        for box in boxes2D:
            x_min, y_min, x_max, y_max = box.coordinates
            x_center = (x_min + x_max) / (2 * self.focal_length)
            y_center = (y_min + y_max) / (2 * self.focal_length)
            z_center = (self.hand_width * focal_length) / (y_max - y_min)
            hands_center.append([x_center, y_center, z_center])
        print(hands_center)
        return hands_center


pipeline = SequentialProcessor()
pipeline.add(DetectMinimalHand(SSD512HandDetection(),
                               MinimalHandPoseEstimation(right_hand=False)))
pipeline.add(pr.UnpackDictionary(['image', 'boxes2D', 'keypoints2D']))
pipeline.add(pr.ControlMap(EstimateHandPosition(focal_length), [1], [1]))
pipeline.add(pr.WrapOutput(['image', 'position']))


player = VideoPlayer((640, 480), pipeline, camera)
player.run()
