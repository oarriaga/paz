import os
import argparse
import numpy as np
import cv2
from pipelines import DetectHumanPose2D
from tensorflow.keras.models import load_model


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

video_capture = cv2.VideoCapture(0)

while True:
    final_heatmaps = None
    ret, image = video_capture.read()
    image = np.array(image).astype(np.uint8)
    detect = DetectHumanPose2D(model, data_info)
    results, scores, image = detect(image)

    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
