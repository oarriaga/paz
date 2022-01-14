import argparse

from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
from pipelines import DetectHandKeypoints
from paz.backend.camera import Camera, VideoPlayer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

use_pretrained = True
HandSegNet = HandSegmentationNet()
HandPoseNet = PoseNet()
HandPosePriorNet = PosePriorNet()
HandViewPointNet = ViewPointNet()

pipeline = DetectHandKeypoints(HandSegNet, HandPoseNet, HandPosePriorNet,
                               HandViewPointNet)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
