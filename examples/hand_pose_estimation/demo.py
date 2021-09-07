import argparse

from HandPoseEstimation import Hand_Segmentation_Net, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet


from detection import DetectHandKeypoints

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer

parser = argparse.ArgumentParser()
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
camera = Camera(args.camera_id)
player = VideoPlayer((256, 256), pipeline, camera)
player.run()
