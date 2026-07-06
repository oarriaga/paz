import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import paz
from paz.datasets.hands import hand_links, link_colors, joint_colors

parser = argparse.ArgumentParser(description="Minimal hand 3D keypoints")
parser.add_argument("-c", "--camera_id", type=int, default=0)
parser.add_argument("--right_hand", action="store_true")
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()

estimate = paz.applications.SSD512MinimalHandPose(right_hand=args.right_hand)
camera = paz.Camera(identifier=args.camera_id)
player = paz.VideoPlayer((args.H, args.W), estimate, camera)
links = np.array(link_colors) / 255
joints = np.array(joint_colors) / 255


def plot_links(ax, keypoints3D):
    for link_arg, (parent, child) in enumerate(hand_links):
        points = np.stack([keypoints3D[parent], keypoints3D[child]], axis=0)
        ax.plot3D(*points.T, c=links[link_arg])


def animate(index):
    outputs = player.step()
    if outputs is None:
        return
    (boxes, keypoints2D, keypoints3D), image = outputs
    image = paz.image.resize_opencv(image, tuple(player.image_size))
    paz.image.show(image, "inference", wait=False)
    if len(keypoints3D) == 0:
        return
    points = np.array(keypoints3D[0])
    plt.cla()
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.scatter3D(*points.T, c=joints)
    plot_links(ax, points)


camera.start()
ax = plt.axes(projection="3d")
ax.view_init(-160, -80)
animation = FuncAnimation(plt.gcf(), animate, interval=1)
plt.tight_layout()
plt.show()
