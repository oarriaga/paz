import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from functools import partial

import numpy as np
import cv2
import jax
import jax.numpy as jp

import paz
import scenes
import codebook

AXIS_COLORS = [(255, 40, 40), (40, 255, 40), (60, 120, 255)]


def rotations_of(poses):
    return jp.stack([jp.asarray(pose)[:3, :3] for pose in poses])


def pairwise_degrees(query_rotations, codebook_rotations):
    query = query_rotations.reshape(len(query_rotations), 9)
    keys = codebook_rotations.reshape(len(codebook_rotations), 9)
    cosine = (query @ keys.T - 1.0) / 2.0
    return jp.degrees(jp.arccos(jp.clip(cosine, -1.0, 1.0)))


def encode_views(encoder, views):
    images = jp.asarray(views, jp.float32) / 255.0
    return codebook.unit_rows(jp.asarray(encoder.predict(images, verbose=0)))


def augment_views(views, key, occlusion_scale=0.25):
    masks = jp.asarray((np.asarray(views) < 230).any(axis=-1))
    randomize = partial(paz.image.randomize_rendered_image,
                        max_radius_scale=occlusion_scale)
    randomize = jax.jit(jax.vmap(randomize, in_axes=(0, 0, 0)))
    keys = jax.random.split(key, len(views))
    return np.asarray(randomize(keys, jp.asarray(views), masks), "uint8")


def retrieve_errors(latents, book_latents, angles):
    args = jp.argmax(latents @ book_latents.T, axis=1)
    return np.asarray(angles[jp.arange(len(args)), args]), np.asarray(args)


def report(name, errors, oracle):
    median = np.median(errors)
    mean = np.mean(errors)
    accuracy = [np.mean(errors < t) for t in (5, 10, 20)]
    print(f"{name:>18}: median {median:5.1f} deg  mean {mean:5.1f} deg  "
          f"acc@5/10/20 {accuracy[0]:.2f}/{accuracy[1]:.2f}/{accuracy[2]:.2f}")
    return median


def project_camera_points(points, focal, half):
    depth = jp.maximum(-points[:, 2], 1e-6)
    column = half + focal * points[:, 0] / depth
    row = half - focal * points[:, 1] / depth
    return np.asarray(jp.stack([column, row], axis=-1)).astype("int32")


def draw_pose(image, pose, focal, half, length, thickness):
    center = jp.asarray(pose)[:3, 3]
    rotation = jp.asarray(pose)[:3, :3]
    tips = center[None, :] + length * rotation.T
    points = project_camera_points(jp.concatenate([center[None], tips]),
                                   focal, half)
    for axis_arg in range(3):
        start, end = tuple(points[0]), tuple(points[axis_arg + 1])
        cv2.line(image, start, end, AXIS_COLORS[axis_arg], thickness)
    return image


def build_montage(clean, augmented, true_poses, pred_poses, focal, size):
    half, length, rows = size / 2.0, 0.4, []
    for index in range(len(clean)):
        overlay = np.ascontiguousarray(clean[index])
        draw_pose(overlay, true_poses[index], focal, half, length, 3)
        draw_pose(overlay, pred_poses[index], focal, half, length, 1)
        triple = [clean[index], augmented[index], overlay]
        rows.append(np.concatenate(triple, axis=1))
    return np.concatenate(rows, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implicit orientation eval")
    parser.add_argument("--weights", default="experiments/aae.weights.h5")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--distance", default=2.5, type=float)
    parser.add_argument("--num_codebook", default=2000, type=int)
    parser.add_argument("--num_test", default=300, type=int)
    parser.add_argument("--output", default="experiments/eval.png")
    args = parser.parse_args()

    model = paz.models.AutoEncoder((args.image_size, args.image_size, 3))
    model.load_weights(args.weights)
    encoder = paz.models.extract_encoder(model)
    mesh = scenes.build_mesh(args.mesh)
    render = scenes.build_renderer(mesh, args.image_size, args.distance)

    book_poses = scenes.random_poses(jax.random.PRNGKey(1), args.num_codebook,
                                     args.distance)
    book = codebook.build_codebook(encoder, render, book_poses)
    book_rotations = rotations_of(book_poses)

    test_poses = scenes.random_poses(jax.random.PRNGKey(7), args.num_test,
                                     args.distance)
    test_views = scenes.render_views(render, test_poses)
    angles = pairwise_degrees(rotations_of(test_poses), book_rotations)
    oracle = np.asarray(jp.min(angles, axis=1))

    clean, _ = retrieve_errors(encode_views(encoder, test_views),
                               book.latents, angles)
    augmented = augment_views(test_views, jax.random.PRNGKey(3))
    noisy, args_noisy = retrieve_errors(encode_views(encoder, augmented),
                                        book.latents, angles)

    print(f"codebook {args.num_codebook} poses, {args.num_test} test views")
    report("oracle floor", oracle, oracle)
    report("clean retrieval", clean, oracle)
    report("augmented retrieval", noisy, oracle)

    focal = float(1.0 / jp.tan(jp.pi / 8.0)) * (args.image_size / 2.0)
    true_poses = [test_poses[index] for index in range(8)]
    pred_poses = [book.poses[int(args_noisy[index])] for index in range(8)]
    montage = build_montage(test_views[:8], augmented[:8], true_poses,
                            pred_poses, focal, args.image_size)
    paz.image.write(args.output, montage)
    print("saved", args.output, "(columns: clean | augmented | "
          "true axes thick / predicted axes thin)")
