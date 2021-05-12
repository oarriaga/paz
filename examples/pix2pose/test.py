import os
import argparse
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import load_model

from paz.abstract.sequence import GeneratingSequencePix2Pose

from model import transformer_loss, loss_color, loss_error
from pipelines import DepthImageGenerator
from scenes import SingleView

description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', type=str, help='Path of 3D OBJ model', default=os.path.join(root_path, 'datasets/035_power_drill/tsdf/textured.obj'))
parser.add_argument('-mp', '--model_path', type=str, help='Path of a trained model')
parser.add_argument('-st', '--steps_per_epoch', default=5, type=int, help='Steps per epoch')
parser.add_argument('-ld', '--image_size', default=128, type=int, help='Size of the side of a square image e.g. 64')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('-id', '--images_directory', type=str, help='Path to directory containing background images', default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-r', '--roll', default=3.14159, type=float, help='Threshold for camera roll in radians')
parser.add_argument('-s', '--shift', default=0.05, type=float, help='Threshold of random shift of camera')
parser.add_argument('-d', '--depth', nargs='+', type=float, default=[0.3, 0.5], help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float, help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float, default=[.5, 30], help='Light intensity from poseur')
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int, help='Flag for full sphere or top half for rendering')
args = parser.parse_args()

print(args.model_path)
model = load_model(args.model_path, custom_objects={'loss_color': loss_color,
                                                    'loss_error': loss_error})


renderer = SingleView(args.obj_path, (args.image_size, args.image_size),
                      args.y_fov, args.depth, args.light, bool(args.top_only),
                      args.roll, args.shift)
image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
processor = DepthImageGenerator(renderer, args.image_size, image_paths, num_occlusions=0)
sequence = GeneratingSequencePix2Pose(processor, model, args.batch_size, args.steps_per_epoch)


sequence_iterator = sequence.__iter__()
for num_batch in range(args.steps_per_epoch):
    batch = next(sequence_iterator)
    predictions = model.predict(batch[0]['input_image'])

    image = ((batch[1]['color_output'][0] + 1)*127.5).astype(np.int32)
    plt.imshow(image)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = list()
    #images = batch[1]['color_output']
    #print(images)
    for i in range(128):
        for j in range(128):
            if image[i, j, 0] > 5 and image[i, j, 1] > 5 and image[i, j, 2] > 5:
                points.append([image[i, j, 0], image[i, j, 1], image[i, j, 2]])
    #plt.imshow(predictions['color_output'][0])
    #plt.show()
    print(len(points))
    points = np.asarray(points)
    print(points.shape)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    plt.show()