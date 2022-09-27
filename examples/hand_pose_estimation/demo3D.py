import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from paz.backend.camera import Camera, VideoPlayer
from paz.applications import SSD512MinimalHandPose
from paz.backend.image import resize_image, show_image
from paz.datasets import MINIMAL_HAND_CONFIG


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = SSD512MinimalHandPose(right_hand=False, offsets=[0.5, 0.5])
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)


def plot_3D_keypoints_link(ax, keypoints3D, link_args, link_orders,
                           link_colors):
    """Plot 3D keypoints links on a plt.axes 3D projection

    # Arguments
        ax: plt.axes object
        keypoints3D: Array, 3D keypoints coordinates
        link_args: Keypoint labels. Dictionary. {'k0':0, 'k1':1, ...}
        link_orders: List of tuple. [('k0', 'k1'),('kl', 'k2'), ...]
        link_colors: Color of each link. List of list
    """
    for pair_arg, pair in enumerate(link_orders):
        color = link_colors[pair_arg]
        point1 = keypoints3D[link_args[pair[0]]]
        point2 = keypoints3D[link_args[pair[1]]]
        points = np.stack([point1, point2], axis=0)
        xs, ys, zs = np.split(points, 3, axis=1)
        ax.plot3D(xs[:, 0], ys[:, 0], zs[:, 0], c=color)


link_orders = MINIMAL_HAND_CONFIG['part_orders']
link_args = MINIMAL_HAND_CONFIG['part_arg']
link_colors = MINIMAL_HAND_CONFIG['part_color']
link_colors = np.array(link_colors) / 255
joint_colors = MINIMAL_HAND_CONFIG['joint_color']
joint_colors = np.array(joint_colors) / 255


def animate(player):
    """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window. Plot the
        3D keypoints on pyplot.
    """
    player.camera.start()
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.set_window_title('Minimal hand 3D plot')

    def wrapped_animate(i):
        output = player.step()
        image = resize_image(output[player.topic], tuple(player.image_size))
        show_image(image, 'inference', wait=False)

        keypoints3D = output['keypoints3D']
        if len(keypoints3D) == 0:
            return
        keypoints3D = keypoints3D[0]  # TAKE ONLY THE FIRST PREDICTION
        xs, ys, zs = np.split(keypoints3D, 3, axis=1)

        plt.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter3D(xs, ys, zs, c=joint_colors)
        plot_3D_keypoints_link(ax, keypoints3D, link_args, link_orders,
                               link_colors)
    return wrapped_animate


animation = FuncAnimation(plt.gcf(), animate(player), interval=1)
plt.tight_layout()
plt.show()
