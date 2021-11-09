import os
import argparse
from pipelines import DetectHumanPose2D
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from tensorflow.keras.models import load_model
from dataset import JOINT_CONFIG, FLIP_CONFIG


parser = argparse.ArgumentParser(description='Test keypoints network')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-m', '--model_weights_path', default='models_weights_tf',
                    help='Path to the model weights')
args = parser.parse_args()

model_path = os.path.join(args.model_weights_path, 'HigherHRNet')
model = load_model(model_path)
print("\n==> Model loaded!\n")

dataset = 'COCO'
data_with_center = False
if data_with_center:
    joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER']
    fliped_joint_order = FLIP_CONFIG[dataset + '_WITH_CENTER']
else:
    joint_order = JOINT_CONFIG[dataset]
    fliped_joint_order = FLIP_CONFIG[dataset]


pipeline = DetectHumanPose2D(model, joint_order, fliped_joint_order,
                             dataset, data_with_center)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
