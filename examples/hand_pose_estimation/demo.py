import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet

from detection import DetectHandKeypoints

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
player = VideoPlayer((256, 256), pipeline, camera)
player.run()
