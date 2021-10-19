import os
import argparse
import numpy as np
import cv2
import processors as pe
from pipelines import DetectHumanPose2D
from tensorflow.keras.models import load_model
from dataset import JOINT_CONFIG, FLIP_CONFIG


parser = argparse.ArgumentParser(description='Test keypoints network')
parser.add_argument('-i', '--image_path', default='image',
                    help='Path to the image')
parser.add_argument('-m', '--model_weights_path', default='models_weights_tf',
                    help='Path to the model weights')
args = parser.parse_args()

image_path = os.path.join(args.image_path, 'image4.jpg')
model_path = os.path.join(args.model_weights_path, 'HigherHRNet')
model = load_model(model_path)
print("\n==> Model loaded!\n")

data_info = {'data': 'COCO',
             'data_with_center': False}


dataset = 'COCO'
data_with_center = False
if data_with_center:
    joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER']
    fliped_joint_order = FLIP_CONFIG[dataset + '_WITH_CENTER']
else:
    joint_order = JOINT_CONFIG[dataset]
    fliped_joint_order = FLIP_CONFIG[dataset]

detect = DetectHumanPose2D(model, joint_order, fliped_joint_order,
                           data_with_center)
draw_skeleton = pe.DrawSkeleton(dataset)
video_capture = cv2.VideoCapture(0)

while True:
    final_heatmaps = None
    ret, image = video_capture.read()
    image = np.array(image).astype(np.uint8)

    joints, scores = detect(image)
    image = draw_skeleton(image, joints)
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# replace it with object_detection's demo.py