import os
import argparse
from pipelines import DetectHumanPose2D
from tensorflow.keras.models import load_model
from paz.backend.image import write_image, load_image


parser = argparse.ArgumentParser(description='Test keypoints network')
parser.add_argument('-i', '--image_path', default='image',
                    help='Path to the image')
parser.add_argument('-m', '--model_weights_path', default='models_weights_tf',
                    help='Path to the model weights')
args = parser.parse_args()

image_path = os.path.join(args.image_path, 'image3.jpg')
model_path = os.path.join(args.model_weights_path, 'HigherHRNet')
model = load_model(model_path)
print("\n==> Model loaded!\n")
image = load_image(image_path)

data_info = {'data': 'COCO',
             'data_with_center': False}
detect = DetectHumanPose2D(model, data_info)
results, scores, image = detect(image)
write_image('output/result.jpg', image)
