import os
import shutil
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from paz.backend.image import load_image
from paz.backend.stereo import remove_outliers
from paz.backend.camera import Camera, VideoPlayer
from paz.pipelines.stereo import StructureFromMotion

parser = argparse.ArgumentParser(description='Structure from motion')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-i', '--images_path', type=str,
                    default='./images', help='Directory for images')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=70, help='Horizontal field of view in degrees')
parser.add_argument('-rt', '--residual_thresh', type=float,
                    default=0.1, help='Residual threshold for RANSAC')
parser.add_argument('-ct', '--correspondence_thresh', type=float,
                    default=0.5, help='Residual threshold for RANSAC for'
                    'correspondence matching')
parser.add_argument('-r', '--match_ratio', type=float,
                    default=0.75, help='Matching ratio for best selection')
parser.add_argument('-b', '--bundle_adjustment', type=bool,
                    default=False, help='Condition to use bundle adjustment')
args = parser.parse_args()


camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline=None, camera=camera)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
camera_intrinsics = camera.intrinsics

print('\n****************************************************************\n')
print('Do you want to start the recording:')
answer = input('yes/no: ')
time.sleep(1)
print('Recording...')
if answer == 'yes':
    print('To stop the recording, press "q" in the capturing window')
    player.record_frames()
    print("\nWaiting for the recording to be saved")
    time.sleep(3)

    shutil.rmtree(args.images_path)
    player.extract_frames_from_video('video.avi')
print('\n****************************************************************\n')
print('Initiating 3D reconstruction')

images = []
image_files = os.listdir(args.images_path)
for filename in image_files:
    image = load_image(os.path.join(args.images_path, filename))
    images.append(image)

detect = StructureFromMotion(camera_intrinsics, least_squares)
inferences = detect(images)


def plot_3D_keypoints(keypoints3D, colors, discard_outliers=True):
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.manager.set_window_title('3D resonstruction')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for arg in range(len(keypoints3D)):
        points3D = keypoints3D[arg]
        color = np.array(colors[arg])
        if discard_outliers:
            points3D, inliers = remove_outliers(keypoints3D[arg], 600)
            color = colors[arg]
            color = np.array(color)[inliers]
        xs, ys, zs = np.split(points3D, 3, axis=1)
        ax.scatter(xs, ys, zs, s=5, c=color/255)
    plt.show()


plot_3D_keypoints(inferences['points3D'], inferences['colors'])
