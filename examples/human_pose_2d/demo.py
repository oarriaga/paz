import argparse
from pipelines import DetectHumanPose2D
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from dataset import JOINT_CONFIG, FLIP_CONFIG


parser = argparse.ArgumentParser(description='Test keypoints network')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

dataset = 'COCO'
data_with_center = False
if data_with_center:
    joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER']
    flipped_joint_order = FLIP_CONFIG[dataset + '_WITH_CENTER']
else:
    joint_order = JOINT_CONFIG[dataset]
    flipped_joint_order = FLIP_CONFIG[dataset]


pipeline = DetectHumanPose2D(joint_order, flipped_joint_order,
                             dataset, data_with_center)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
