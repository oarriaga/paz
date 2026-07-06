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
    H, W = size
    images = np.empty((num_views, H, W, 3), "uint8")
    coordinates = np.empty((num_views, H, W, 3), "uint8")
    masks = np.empty((num_views, H, W), "uint8")
    for view, key in enumerate(jax.random.split(jax.random.PRNGKey(seed), num_views)):  # fmt: skip
        pose = scenes.sample_pose(key, distance)
        nocs, mask = render_coordinates(pose)
        images[view] = np.asarray(render_image(pose))
        coordinates[view] = (np.asarray(nocs) * 255).astype("uint8")
        masks[view] = (np.asarray(mask) > 0.5).astype("uint8")
    np.savez(cache, images=images, coordinates=coordinates, masks=masks)
    return images, coordinates, masks


def build_evaluation(mesh, size, distance, y_FOV, chunk_size, tiles, camera,
                     extents, num_samples, num_points, seed):
    render_image = scenes.build_image_renderer(
        mesh, size, np.mean(distance), y_FOV, chunk_size, tiles)
    render_coordinates = scenes.build_coordinate_renderer(
        mesh, size, y_FOV, chunk_size)
    randomize = jax.jit(paz.image.randomize_rendered_image)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    inputs, poses_true = [], []
    for key in keys:
        pose = scenes.sample_pose(key, distance)
        image = render_image(pose)
        nocs, mask = render_coordinates(pose)
        nocs, mask = np.asarray(nocs), np.asarray(mask) > 0.5
        pose_true = pipeline.solve_pose_from_nocs(nocs, mask, extents, camera)
        if pose_true is None:
            continue
        randomized = np.asarray(randomize(key, image, mask.astype("float32")))
        inputs.append(np.clip(randomized, 0, 255) / 255.0)
        poses_true.append(pose_true)
    points3D = np.asarray(mesh.vertices)
    choice = np.random.RandomState(0).choice(len(points3D), num_points, False)
    return inputs, poses_true, points3D[choice]


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
    parser.add_argument("--num_views", default=40000, type=int)
    parser.add_argument("--num_backgrounds", default=4000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--beta", default=3.0, type=float)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--eval_samples", default=50, type=int)
    parser.add_argument("--eval_period", default=5, type=int)
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    size = (args.image_size, args.image_size)
    mesh_path = args.mesh or scenes.download_power_drill()
    mesh = scenes.build_mesh(mesh_path, args.target_faces)
    extents = np.asarray(scenes.object_extents(mesh))
    images, coordinates, masks = render_dataset(
        mesh, args.num_views, size, args.distance, args.y_FOV,
        args.chunk_size, tuple(args.tiles), args.root, seed=0)

    backgrounds = pipeline.load_backgrounds(
        args.backgrounds, args.image_size, args.num_backgrounds)
    sequence = pipeline.Pix2PoseSequence(
        images, coordinates, masks, args.batch_size, backgrounds)

    camera = pipeline.build_camera(args.y_FOV, size)
    eval_inputs, eval_poses, eval_points = build_evaluation(
        mesh, size, args.distance, args.y_FOV, args.chunk_size,
        tuple(args.tiles), camera, extents, args.eval_samples, 512, seed=999)
    diameter = paz.evaluation.compute_object_diameter(eval_points)

    def predict_pose(model, image):
        prediction = model(jp.expand_dims(image, 0))
        nocs = np.asarray(jp.squeeze(prediction, 0))
        mask = nocs.sum(-1) > 0.15
        return pipeline.solve_pose_from_nocs(nocs, mask, extents, camera)

    model = paz.models.UNET_VGG16(3, (*size, 3), freeze_backbone=True)
    model.compile(keras.optimizers.Adam(1e-3),
                  paz.losses.WeightedReconstruction(args.beta))
    weights = os.path.join(args.root, "UNET-VGG16_POWERDRILL.weights.h5")
    callbacks = [
        paz.callbacks.EvaluatePose(eval_inputs, eval_poses, eval_points,
                                   diameter, predict_pose, args.eval_period),
        keras.callbacks.EarlyStopping("loss", patience=args.patience,
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(os.path.join(args.root, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_weights_only=True),
    ]
    model.fit(sequence, epochs=args.epochs, callbacks=callbacks)
