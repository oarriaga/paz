import argparse

from HandPoseEstimation import Hand_Segmentation_Net, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
<<<<<<< HEAD
from hand_keypoints_loader import RenderedHandLoader
=======

>>>>>>> Working code update

from detection import DetectHandKeypoints

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer

parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument('--data_path', type=str, help='Path to dataset')
=======
>>>>>>> Working code update
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

<<<<<<< HEAD
#data_manager = RenderedHandLoader(args.data_path, 'val')
#data = data_manager.load_data()

=======
>>>>>>> Working code update

use_pretrained = True
HandSegNet = Hand_Segmentation_Net(load_pretrained=use_pretrained)
HandPoseNet = PoseNet(load_pretrained=use_pretrained)
HandPosePriorNet = PosePriorNet(load_pretrained=use_pretrained)
HandViewPointNet = ViewPointNet(load_pretrained=use_pretrained)


pipeline = DetectHandKeypoints(HandSegNet, HandPoseNet, HandPosePriorNet,
                               HandViewPointNet)
camera = Camera(args.camera_id)
<<<<<<< HEAD
player = VideoPlayer((640, 480), pipeline, camera)
=======
player = VideoPlayer((256, 256), pipeline, camera)
>>>>>>> Working code update
player.run()
