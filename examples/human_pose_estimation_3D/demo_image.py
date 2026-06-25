import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import paz
from viz import show3Dpose

parser = argparse.ArgumentParser(description="Human pose 3D + 6D on an image")
parser.add_argument("--image", required=True, help="RGB image path")
parser.add_argument("--HFOV", default=70, type=int)
parser.add_argument("--output", default="human_pose_6D.jpg")
parser.add_argument("--output3D", default="human_pose_3D.png")
args = parser.parse_args()

image = paz.image.load(args.image)
height, width = paz.image.get_size(image)
camera = paz.Camera()
intrinsics = camera.intrinsics_from_HFOV(args.HFOV, (height, width))

estimate = paz.applications.EstimateHumanPose(least_squares, intrinsics)
(keypoints2D, poses3D, pose6D), drawn = estimate(image)
paz.image.write(args.output, drawn)

axes = plt.figure().add_subplot(projection="3d")
axes.view_init(-160, -80)
show3Dpose(np.array(poses3D), axes)
plt.savefig(args.output3D, dpi=80, bbox_inches="tight")
