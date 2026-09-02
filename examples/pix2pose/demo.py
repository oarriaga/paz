import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jp

import paz
from paz.poses import solve_PnP_RANSAC, project_points3D
from paz.pinhole import build_cube_points3D
import scenes

Camera = namedtuple("Camera", ["intrinsics", "distortion"])

parser = argparse.ArgumentParser(description="PIX2POSE 6D pose from coordinates")
parser.add_argument("--weights", default=None, help="UNET weights; GT if unset")
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--distance", nargs=2, default=[0.35, 0.45], type=float)
parser.add_argument("--y_FOV", default=float(jp.pi / 4), type=float)
parser.add_argument("--chunk_size", default=1024 * 4, type=int)
parser.add_argument("--tiles", nargs=2, default=[2, 2], type=int)
parser.add_argument("--target_faces", default=20000, type=int)
parser.add_argument("--max_points", default=1500, type=int)
parser.add_argument("--seed", default=7, type=int)
parser.add_argument("--output", default="pix2pose_pose.png")
args = parser.parse_args()

size = (args.image_size, args.image_size)
H, W = size
mesh = scenes.build_mesh(scenes.download_power_drill(), args.target_faces)
extents = np.asarray(scenes.object_extents(mesh))

render_image = scenes.build_image_renderer(
    mesh, size, np.mean(args.distance), args.y_FOV, args.chunk_size, tuple(args.tiles))  # fmt: skip
render_coordinates = scenes.build_coordinate_renderer(
    mesh, size, args.y_FOV)

pose = scenes.sample_pose(jax.random.PRNGKey(args.seed), args.distance)
image = np.asarray(render_image(pose))
nocs, mask = render_coordinates(pose)
nocs, mask = np.asarray(nocs), np.asarray(mask)

if args.weights is not None:
    model = paz.models.UNET_VGG16(3, (*size, 3))
    model.load_weights(args.weights)
    prediction = model(jp.expand_dims(image / 255.0, 0))
    nocs = np.asarray(jp.squeeze(prediction, 0))
    mask = (nocs.sum(-1) > 0.05).astype("float32")

# Camera intrinsics matching the renderer's y_FOV pinhole projection.
focal = (1.0 / np.tan(args.y_FOV / 2.0)) * (H / 2.0)
intrinsics = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])
camera = Camera(intrinsics, np.zeros((4, 1)))

rows, cols = np.nonzero(mask > 0.5)
points2D = np.stack([cols, rows], axis=1).astype("float64")
points3D = extents * (nocs[rows, cols] - 0.5)      # invert NOCS to object frame
if len(points3D) > args.max_points:
    choice = np.random.RandomState(0).choice(len(points3D), args.max_points, False)  # fmt: skip
    points2D, points3D = points2D[choice], points3D[choice]

pose6D = solve_PnP_RANSAC(points2D, points3D, camera)
assert pose6D is not None, "PnP-RANSAC failed"

reprojected = np.asarray(project_points3D(points3D, pose6D, camera))
reprojection_error = np.mean(np.linalg.norm(reprojected - points2D, axis=1))
print(f"correspondences: {len(points3D)} | "
      f"mean reprojection error: {reprojection_error:.2f} px")

cube = paz.to_numpy(build_cube_points3D(*extents))
drawn = paz.applications.pose_estimators.draw_boxes3D(
    image.copy(), [pose6D], cube, camera, paz.draw.GREEN, 2, 3)
paz.image.write(args.output, drawn)
print(f"wrote {args.output}")
