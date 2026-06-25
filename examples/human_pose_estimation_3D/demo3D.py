import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import least_squares

import paz
from viz import show3Dpose

parser = argparse.ArgumentParser(description="Live 3D human-pose visualization")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--HFOV", default=70, type=int)
args = parser.parse_args()

camera = paz.Camera(identifier=args.camera)
intrinsics = camera.intrinsics_from_HFOV(args.HFOV, (args.H, args.W))
pipeline = paz.applications.EstimateHumanPose(least_squares, intrinsics)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)


def animate(player):
    player.camera.start()
    axes = plt.axes(projection="3d")
    axes.view_init(-160, -80)

    def update(frame):
        output = player.step()
        paz.image.show(output[player.topic], "inference", wait=False)
        keypoints2D, poses3D, pose6D = output[0]
        if len(keypoints2D) == 0:
            return
        plt.cla()
        show3Dpose(np.array(poses3D), axes)

    return update


animation = FuncAnimation(plt.gcf(), animate(player), interval=1)
plt.tight_layout()
plt.show()
