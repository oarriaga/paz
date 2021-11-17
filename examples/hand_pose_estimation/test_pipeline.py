import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
from pipelines import DetectHandKeypoints
from paz.backend.image.opencv_image import load_image, show_image

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

img = load_image('./sample.jpg')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))

img = load_image('./images/00149.png')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))

img = load_image('./images/img5.png')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))