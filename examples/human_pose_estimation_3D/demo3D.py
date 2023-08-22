import os
from paz.backend.camera import Camera
from paz.backend.image import load_image
from scipy.optimize import least_squares
from tensorflow.keras.utils import get_file
from paz.pipelines import EstimateHumanPose
from paz.processors import OptimizeHumanPose3D
from paz.datasets.human36m import args_to_joints3D
from viz import visualize, show3Dpose

import argparse
from paz.backend.camera import VideoPlayer
import matplotlib.pyplot as plt
from paz.backend.image import resize_image, show_image
import numpy as np
from matplotlib.animation import FuncAnimation


parser = argparse.ArgumentParser(description='Human3D visualization')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=(640, 480))
pipeline = EstimateHumanPose()
optimize = OptimizeHumanPose3D(args_to_joints3D,
                               least_squares, camera.intrinsics)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)


def animate(player):
    """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window. Plot the
        3D keypoints on pyplot.
    """
    player.camera.start()
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.manager.set_window_title('Human pose visualization')

    def wrapped_animate(i):
        output = player.step()
        image = resize_image(output[player.topic], tuple(player.image_size))
        show_image(image, 'inference', wait=False)

        keypoints2D = output['keypoints2D']
        keypoints3D = output['keypoints3D']
        pose = optimize(keypoints3D, keypoints2D)
        if len(keypoints3D) == 0:
            return
        keypoints3D = keypoints3D[0]  # TAKE ONLY THE FIRST PREDICTION
        xs, ys, zs = np.split(keypoints3D, 3, axis=1)

        plt.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.scatter3D(xs, ys, zs)
        show3Dpose(pose[2], ax)
    return wrapped_animate


animation = FuncAnimation(plt.gcf(), animate(player), interval=1)
plt.tight_layout()
plt.show()
