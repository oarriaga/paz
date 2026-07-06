import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jp

import paz
from paz.applications.pose_estimators import solve_PnP_RANSAC
from paz.applications.pose_estimators import project_points3D, build_cube_corners
from paz.applications.pose_estimators import draw_boxes3D
import scenes
import pipeline

Camera = namedtuple("Camera", ["intrinsics", "distortion"])

parser = argparse.ArgumentParser(description="Validate PIX2POSE performance")
parser.add_argument("--weights", required=True)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--distance", nargs=2, default=[0.35, 0.45], type=float)
parser.add_argument("--y_FOV", default=float(jp.pi / 4), type=float)
parser.add_argument("--chunk_size", default=1024 * 4, type=int)
parser.add_argument("--tiles", nargs=2, default=[2, 2], type=int)
parser.add_argument("--target_faces", default=20000, type=int)
parser.add_argument("--num_samples", default=30, type=int)
parser.add_argument("--num_show", default=6, type=int)
parser.add_argument("--max_points", default=1500, type=int)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("--output", default="pix2pose_validation.png")
args = parser.parse_args()

size = (args.image_size, args.image_size)
H, W = size
mesh = scenes.build_mesh(scenes.download_power_drill(), args.target_faces)
extents = np.asarray(scenes.object_extents(mesh))
render_image = scenes.build_image_renderer(
    mesh, size, np.mean(args.distance), args.y_FOV, args.chunk_size, tuple(args.tiles))  # fmt: skip
render_coordinates = scenes.build_coordinate_renderer(
    mesh, size, args.y_FOV, args.chunk_size)

model = paz.models.UNET_VGG16(3, (*size, 3))
model.load_weights(args.weights)

focal = (1.0 / np.tan(args.y_FOV / 2.0)) * (H / 2.0)
intrinsics = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])
camera = Camera(intrinsics, np.zeros((4, 1)))
randomize = jax.jit(jax.vmap(paz.image.randomize_rendered_image,
                             in_axes=(0, 0, 0, None)))
backgrounds = jp.asarray(pipeline.load_backgrounds(os.path.expanduser(
    "~/.keras/paz/datasets/voc-backgrounds"), args.image_size, 400))

keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_samples)
nocs_errors, pose_errors, solved = [], [], 0
cells = []
for arg, key in enumerate(keys):
    pose = scenes.sample_pose(key, args.distance)
    image = np.asarray(render_image(pose))
    nocs_true, mask = render_coordinates(pose)
    nocs_true, mask = np.asarray(nocs_true), np.asarray(mask) > 0.5
    model_input = randomize(key[None], image[None], mask[None].astype("float32"),
                            backgrounds)[0] / 255.0
    nocs_pred = np.asarray(jp.squeeze(model(model_input[None]), 0))

    nocs_errors.append(np.abs(nocs_pred[mask] - nocs_true[mask]).mean())

    predicted_mask = nocs_pred.sum(-1) > 0.15
    rows, cols = np.nonzero(predicted_mask)
    points2D = np.stack([cols, rows], axis=1).astype("float64")
    points3D = extents * (nocs_pred[rows, cols] - 0.5)
    if len(points3D) > args.max_points:
        choice = np.random.RandomState(0).choice(len(points3D), args.max_points, False)  # fmt: skip
        points2D, points3D = points2D[choice], points3D[choice]

    pose6D = solve_PnP_RANSAC(points2D, points3D, camera)
    if pose6D is not None:
        solved += 1
        rows_t, cols_t = np.nonzero(mask)          # ground-truth correspondences
        truth2D = np.stack([cols_t, rows_t], axis=1).astype("float64")
        truth3D = extents * (nocs_true[rows_t, cols_t] - 0.5)
        projected = np.asarray(project_points3D(truth3D, pose6D, camera))
        pose_errors.append(np.linalg.norm(projected - truth2D, axis=1).mean())

    if arg < args.num_show and pose6D is not None:
        cube = paz.to_numpy(build_cube_corners(*extents))
        overlay = draw_boxes3D(image.copy(), [pose6D], cube, camera, paz.draw.GREEN, 2, 3)  # fmt: skip
        row = [image, (nocs_true * 255).astype("uint8"),
               (np.clip(nocs_pred, 0, 1) * 255).astype("uint8"), overlay]
        row = [np.pad(c, ((0, 0), (3, 3), (0, 0)), constant_values=255) for c in row]  # fmt: skip
        cells.append(np.concatenate(row, axis=1))

print(f"samples: {args.num_samples} | PnP solved: {solved}/{args.num_samples}")
print(f"NOCS foreground MAE (pred vs GT, [0,1]): {np.mean(nocs_errors):.4f}")
print(f"pose reprojection error vs GT correspondences: "
      f"{np.mean(pose_errors):.2f} px (median {np.median(pose_errors):.2f})")

sheet = np.concatenate([np.pad(r, ((3, 3), (0, 0), (0, 0)), constant_values=255)
                        for r in cells], axis=0)
paz.image.write(args.output, sheet)
print("columns: clean render | GT NOCS | predicted NOCS | predicted pose box")
print(f"wrote {args.output}")
