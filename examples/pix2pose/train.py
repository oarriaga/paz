import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import numpy as np
import jax
import jax.numpy as jp
import keras

import paz
import scenes
import pipeline


def render_dataset(mesh, num_views, size, distance, y_FOV, chunk_size, tiles,
                   root, seed):
    cache = os.path.join(root, "views.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return data["images"], data["coordinates"], data["masks"]
    render_image = scenes.build_image_renderer(
        mesh, size, np.mean(distance), y_FOV, chunk_size, tiles)
    render_coordinates = scenes.build_coordinate_renderer(
        mesh, size, y_FOV, chunk_size)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_views)
    poses = [scenes.sample_pose(key, distance) for key in keys]
    views = pipeline.render_views(render_image, render_coordinates, poses)
    np.savez(cache, images=views[0], coordinates=views[1], masks=views[2])
    return views


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PIX2POSE power drill")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--backgrounds", default=os.path.expanduser(
        "~/.keras/paz/datasets/voc-backgrounds"))
    parser.add_argument("--root", default="experiments/power_drill")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--distance", nargs=2, default=[0.35, 0.45], type=float)
    parser.add_argument("--y_FOV", default=float(jp.pi / 4), type=float)
    parser.add_argument("--chunk_size", default=1024 * 4, type=int)
    parser.add_argument("--tiles", nargs=2, default=[2, 2], type=int)
    parser.add_argument("--target_faces", default=20000, type=int)
    parser.add_argument("--num_views", default=10000, type=int)
    parser.add_argument("--num_backgrounds", default=2000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--beta", default=3.0, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    size = (args.image_size, args.image_size)
    mesh_path = args.mesh or scenes.download_power_drill()
    mesh = scenes.build_mesh(mesh_path, args.target_faces)
    images, coordinates, masks = render_dataset(
        mesh, args.num_views, size, args.distance, args.y_FOV,
        args.chunk_size, tuple(args.tiles), args.root, seed=0)

    backgrounds = pipeline.load_backgrounds(
        args.backgrounds, args.image_size, args.num_backgrounds)
    sequence = pipeline.Pix2PoseSequence(
        images, coordinates, masks, args.batch_size, backgrounds)

    model = paz.models.UNET_VGG16(3, (*size, 3), freeze_backbone=True)
    model.compile(keras.optimizers.Adam(1e-3),
                  paz.losses.WeightedReconstruction(args.beta))
    weights = os.path.join(args.root, "UNET-VGG16_POWERDRILL.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.root, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_weights_only=True),
    ]
    model.fit(sequence, epochs=args.epochs, callbacks=callbacks)
