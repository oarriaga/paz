import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import time
import numpy as np
import jax
import jax.numpy as jp

import paz
import scenes
import pipeline

parser = argparse.ArgumentParser(description="Visualize PIX2POSE data pipeline")
parser.add_argument("--num_samples", default=5, type=int)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--distance", nargs=2, default=[0.35, 0.45], type=float)
parser.add_argument("--y_FOV", default=float(jp.pi / 4), type=float)
parser.add_argument("--chunk_size", default=1024 * 4, type=int)
parser.add_argument("--tiles", nargs=2, default=[2, 2], type=int)
parser.add_argument("--target_faces", default=20000, type=int)
parser.add_argument("--num_backgrounds", default=200, type=int)
parser.add_argument("--backgrounds", default=os.path.expanduser(
    "~/.keras/paz/datasets/voc-backgrounds"))
parser.add_argument("--output", default="pix2pose_pipeline.png")
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

size = (args.image_size, args.image_size)
mesh = scenes.build_mesh(scenes.download_power_drill(), args.target_faces)
extents = np.asarray(scenes.object_extents(mesh))
print(f"drill: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
print(f"object extents (m): {np.round(extents, 4)}")

render_image = scenes.build_image_renderer(
    mesh, size, np.mean(args.distance), args.y_FOV, args.chunk_size, tuple(args.tiles))  # fmt: skip
render_coordinates = scenes.build_coordinate_renderer(
    mesh, size, args.y_FOV, args.chunk_size)

keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_samples)
poses = [scenes.sample_pose(key, args.distance) for key in keys]

start = time.time()
images, coordinates, masks = pipeline.render_views(
    render_image, render_coordinates, poses)
render_ms = 1000 * (time.time() - start) / args.num_samples
print(f"render: {render_ms:.0f} ms/sample (RGB + coordinates), {size} px")

backgrounds = pipeline.load_backgrounds(
    args.backgrounds, args.image_size, args.num_backgrounds)
randomize = jax.jit(jax.vmap(paz.image.randomize_rendered_image,
                             in_axes=(0, 0, 0, None)))
inputs = np.asarray(randomize(keys, images, masks, backgrounds))

# Validate the label is exactly what training expects.
in_mask = coordinates[masks > 0.5]
assert coordinates[masks < 0.5].max() == 0.0, "coordinates leak outside mask"
assert in_mask.min() >= 0.0 and in_mask.max() <= 1.0, "coordinates outside [0,1]"
assert not np.allclose(inputs, images), "randomization did not run"
print(f"coordinates in-mask range [{in_mask.min():.3f}, {in_mask.max():.3f}], "
      f"0 outside mask; mask coverage {masks.mean():.3f}")


def to_rgb(image):
    return np.repeat(image[..., None], 3, axis=-1)


def contact_sheet(columns, pad=4):
    rows = []
    for row in columns:
        cells = [np.pad(cell, ((0, 0), (pad, pad), (0, 0)), constant_values=255)
                 for cell in row]
        rows.append(np.concatenate(cells, axis=1))
    rows = [np.pad(row, ((pad, pad), (0, 0), (0, 0)), constant_values=255)
            for row in rows]
    return np.concatenate(rows, axis=0)


columns = []
for arg in range(args.num_samples):
    clean = images[arg]
    randomized = np.clip(inputs[arg], 0, 255).astype("uint8")
    nocs = (coordinates[arg] * 255).astype("uint8")
    mask = to_rgb((masks[arg] * 255).astype("uint8"))
    columns.append([clean, randomized, nocs, mask])

sheet = contact_sheet(columns)
paz.image.write(args.output, sheet)
print("columns: clean render | randomized input | NOCS label | mask")
print(f"wrote {args.output}  ({sheet.shape[1]}x{sheet.shape[0]})")
