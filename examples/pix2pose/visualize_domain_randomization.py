import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import numpy as np
import jax
import jax.numpy as jp

import paz
import scenes
import pipeline

parser = argparse.ArgumentParser(description="Corroborate PIX2POSE domain rand.")
parser.add_argument("--num_poses", default=4, type=int)
parser.add_argument("--num_randomizations", default=4, type=int)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--distance", nargs=2, default=[0.35, 0.45], type=float)
parser.add_argument("--y_FOV", default=float(jp.pi / 4), type=float)
parser.add_argument("--chunk_size", default=1024 * 4, type=int)
parser.add_argument("--tiles", nargs=2, default=[2, 2], type=int)
parser.add_argument("--target_faces", default=20000, type=int)
parser.add_argument("--num_backgrounds", default=400, type=int)
parser.add_argument("--backgrounds", default=os.path.expanduser(
    "~/.keras/paz/datasets/voc-backgrounds"))
parser.add_argument("--output", default="pix2pose_domain_randomization.png")
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

size = (args.image_size, args.image_size)
mesh = scenes.build_mesh(scenes.download_power_drill(), args.target_faces)
render_image = scenes.build_image_renderer(
    mesh, size, np.mean(args.distance), args.y_FOV, args.chunk_size, tuple(args.tiles))  # fmt: skip
render_coordinates = scenes.build_coordinate_renderer(
    mesh, size, args.y_FOV)
backgrounds = jp.asarray(pipeline.load_backgrounds(
    args.backgrounds, args.image_size, args.num_backgrounds))
randomize = jax.jit(paz.image.randomize_rendered_image)


def mask_boundary_overlay(image, mask):
    m = mask > 0.5
    shifted = (np.roll(m, 1, 0) & np.roll(m, -1, 0) &
               np.roll(m, 1, 1) & np.roll(m, -1, 1))
    edge = m & ~shifted
    overlay = image.copy()
    overlay[edge] = np.array([0, 255, 0], "uint8")
    return overlay


def to_uint8(image):
    return np.clip(np.asarray(image), 0, 255).astype("uint8")


rows = []
pose_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_poses)
for pose_key in pose_keys:
    pose = scenes.sample_pose(pose_key, args.distance)
    clean = np.asarray(render_image(pose))
    nocs, mask = render_coordinates(pose)
    nocs, mask = np.asarray(nocs), np.asarray(mask)
    randomizations = []
    for aug_key in jax.random.split(pose_key, args.num_randomizations):
        image = randomize(aug_key, clean, mask, backgrounds)
        randomizations.append(to_uint8(image))
    row = ([clean] + randomizations +
           [(nocs * 255).astype("uint8"),
            mask_boundary_overlay(clean, mask)])
    row = [np.pad(c, ((0, 0), (3, 3), (0, 0)), constant_values=255) for c in row]
    rows.append(np.concatenate(row, axis=1))

sheet = np.concatenate([np.pad(r, ((3, 3), (0, 0), (0, 0)), constant_values=255)
                        for r in rows], axis=0)
paz.image.write(args.output, sheet)
labels = "clean | " + " | ".join(f"rand{arg + 1}" for arg in range(args.num_randomizations))  # fmt: skip
print(f"columns: {labels} | NOCS label | label boundary on render")
print(f"wrote {args.output}  ({sheet.shape[1]}x{sheet.shape[0]})")
