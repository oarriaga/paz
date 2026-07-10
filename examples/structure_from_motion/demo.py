import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import jax
import matplotlib.pyplot as plt
import paz

import backend
import pipeline

parser = argparse.ArgumentParser(description="Structure from motion")
parser.add_argument("--images_path", type=str, default="datasets/images",
                    help="Directory with an ordered sequence of images")
parser.add_argument("--HFOV", type=float, default=70,
                    help="Horizontal field of view in degrees")
parser.add_argument("--match_ratio", type=float, default=0.75)
parser.add_argument("--residual_thresh", type=float, default=0.5,
                    help="Sampson-distance RANSAC threshold between frames")
parser.add_argument("--correspondence_thresh", type=float, default=0.5,
                    help="Sampson-distance RANSAC threshold for PnP tracking")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def plot_3D_keypoints(points3D_list, colors_list, outlier_thresh=80):
    axis = plt.axes(projection="3d")
    axis.view_init(-160, -80)
    axis.set_xlabel("X"), axis.set_ylabel("Y"), axis.set_zlabel("Z")
    for points3D, colors in zip(points3D_list, colors_list):
        points3D, inliers = backend.remove_outliers(points3D, outlier_thresh)
        axis.scatter(*points3D.T, s=5, c=colors[inliers] / 255.0)
    plt.show()


image_names = sorted(os.listdir(args.images_path))
images = [paz.to_numpy(paz.image.load(os.path.join(args.images_path, name)))
         for name in image_names]
H, W = images[0].shape[:2]
camera_intrinsics = paz.pinhole.intrinsics_from_HFOV(H, W, args.HFOV)
camera_intrinsics = paz.to_numpy(camera_intrinsics)

key = jax.random.PRNGKey(args.seed)
reconstruction = pipeline.reconstruct_scene(
    images, camera_intrinsics, key, args.match_ratio, args.residual_thresh,
    args.correspondence_thresh)
plot_3D_keypoints(reconstruction.points3D, reconstruction.colors)
