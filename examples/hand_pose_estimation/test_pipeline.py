import argparse
import matplotlib.pyplot as plt
import numpy as np

from HandPoseEstimation import Hand_Segmentation_Net, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet

from detection import DetectHandKeypoints
from utils import load

from paz.backend.image.opencv_image import load_image, show_image

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

use_pretrained = True
HandSegNet = Hand_Segmentation_Net(load_pretrained=use_pretrained)
HandPoseNet = PoseNet(load_pretrained=use_pretrained)
HandPosePriorNet = PosePriorNet(load_pretrained=use_pretrained)
HandViewPointNet = ViewPointNet(load_pretrained=use_pretrained)

pipeline = DetectHandKeypoints(HandSegNet, HandPoseNet, HandPosePriorNet,
                               HandViewPointNet)

img = load_image('./sample.jpg')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))

