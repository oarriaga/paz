import os
import pickle
import argparse
from pathlib import Path
from glob import glob

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jp
import keras
import paz


def build_render(image_shape, y_FOV, camera_origin, shadows):
    target = jp.array([0.0, 0.0, 0.0])
    upward = jp.array([0.0, 1.0, 0.0])
    pose = paz.SE3.view_transform(jp.array(camera_origin), target, upward)
    rays = paz.graphics.camera.build_rays(image_shape, y_FOV, pose)
    return paz.partial(
        paz.graphics.render,
        image_shape=image_shape,
        world_to_camera=pose,
        rays=rays,
        shadows=shadows,
    )


def render_dataset(render, shapes, floor, lights):
    images = []
    for shape in shapes:
        scene = paz.graphics.Scene([shape, floor])
        image, _ = render(scene=scene, mask=None, lights=lights)
        images.append(jp.clip(image, 0.0, 1.0))
    return np.array(images)


def build_feature_backbone(model_name, weights):
    model_kwargs = {"include_top": False, "weights": weights}
    if model_name == "VGG16":
        model = keras.applications.VGG16(**model_kwargs)
        preprocess = keras.applications.vgg16.preprocess_input
        layer_names = []
        for block in range(1, 6):
            num_convs = 2 if block <= 2 else 3
            for conv in range(1, num_convs + 1):
                layer_names.append(f"block{block}_conv{conv}")
        return model, preprocess, layer_names
    if model_name == "TINYCONVNEXT":
        model = keras.applications.convnext.ConvNeXtTiny(**model_kwargs)
        preprocess = keras.applications.convnext.preprocess_input
        layer_names = ["stem"] + [f"stage_{i}_block_0" for i in range(4)]
        return model, preprocess, layer_names
    raise ValueError(f"Unknown model '{model_name}'")


def build_feature_model(model, layer_name):
    output = model.get_layer(layer_name).output
    return keras.Model(inputs=model.input, outputs=output)


def extract_image_features(image, model, preprocess):
    image = preprocess(image)
    image = jp.expand_dims(image, axis=0)
    featuremap = model(image, training=False)
    featuremap = np.array(featuremap)[0]
    if featuremap.ndim == 3:  # Move channel (H, W, C) to (C, H, W)
        featuremap = np.transpose(featuremap, (2, 0, 1))
    return featuremap


def denormalize_image(image):
    return image * 255.0


def extract_features(images, model, preprocess):
    features = []
    for image in images:
        image = denormalize_image(image)
        featuremap = extract_image_features(image, model, preprocess)
        features.append(featuremap)
    return np.array(features)


def process_features(features):
    x = features - np.mean(features, axis=(1, 2), keepdims=True)
    x = x / (np.std(features, axis=(1, 2), keepdims=True) + 1e-5)
    x = x * 0.15
    x = x + 0.50
    return np.clip(x, 0, 1)


def compute_feature_error(true_features, pred_features):
    return np.mean((true_features - pred_features) ** 2, axis=(2, 3))


def compute_sparsity(features, threshold=1e-6):
    sparsity = np.mean(np.abs(features) < threshold)
    magnitude = np.mean(np.abs(features))
    return sparsity, magnitude


def flatten_layer_errors(layer_to_results):
    layer_to_errors, layer_to_features = [], []
    for featuremap, errors in layer_to_results:
        layer_to_errors.append(errors)
        layer_to_features.append(featuremap)
    layer_to_errors = np.array(layer_to_errors)
    errors = layer_to_errors.reshape(-1)
    return errors, layer_to_features, layer_to_errors.shape[1]


def find_top_k_indices(errors, num_features, top_k):
    lower_error_args = np.argsort(errors)
    row_args = lower_error_args // num_features
    col_args = lower_error_args - (row_args * num_features)
    return row_args[:top_k], col_args[:top_k]


def build_invariances_dict(row_args, col_args, layer_names, layer_to_features):
    invariances = {}
    for arg, (row_arg, col_arg) in enumerate(zip(row_args, col_args)):
        model_layer = layer_names[row_arg]
        feature_arg = int(layer_to_features[row_arg][col_arg])
        invariances[arg] = {"layer": model_layer, "featuremap": feature_arg}
    return invariances


def extract_invariances(layer_to_results, layer_names, top_k=5):
    layer_errors = flatten_layer_errors(layer_to_results)
    errors, layer_to_features, num_features = layer_errors
    args = find_top_k_indices(errors, num_features, top_k)
    return build_invariances_dict(*args, layer_names, layer_to_features)


def write_image(image, directory, filename):
    filepath = Path(directory) / filename
    image = paz.image.denormalize(jp.clip(image, 0.0, 1.0))
    paz.image.write(str(filepath), image)


def write_featuremap(featuremap, directory, filename):
    featuremap = jp.array(featuremap)
    if featuremap.ndim == 2:
        featuremap = jp.expand_dims(featuremap, axis=-1)
    if featuremap.shape[-1] == 1:
        featuremap = jp.repeat(featuremap, 3, axis=-1)
    filepath = Path(directory) / filename
    image = paz.image.denormalize(jp.clip(featuremap, 0.0, 1.0))
    paz.image.write(str(filepath), image)


def resolve_path(path, script_directory):
    path = Path(path)
    return path if path.is_absolute() else script_directory / path


def find_dataset_name(root_path, dataset_name, parameters_wildcard):
    wildcard = str(root_path / dataset_name / parameters_wildcard)
    if len(glob(wildcard)) == 0:
        candidates = [path for path in root_path.iterdir() if path.is_dir()]
        if len(candidates) == 1:
            return candidates[0].name, wildcard
    return dataset_name, wildcard


def setup_output_directory(root_path, dataset_name, model_name, label, args):
    full_label = "-".join([model_name, label])
    root = paz.directory.make_timestamped(root_path / dataset_name, full_label)
    paz.file.write_json(args.__dict__, Path(root) / args.parameters_filename)
    return root


def load_optimization_metadata(wildcard, parameters_filename):
    metadata = paz.file.load_latest(wildcard, parameters_filename)
    optimization_directory = paz.directory.find_latest(wildcard)
    return metadata, optimization_directory


def compute_image_shape(dataset_name, viewport_factor):
    dataset_metadata = paz.datasets.fsclvr.parse_metadata(dataset_name)
    H, W = dataset_metadata["image_shape"]
    image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
    camera_origin = jp.array(dataset_metadata["camera_origin"])
    y_FOV = dataset_metadata["y_FOV"]
    return image_shape, camera_origin, y_FOV


def load_shapes(shapes_directory):
    shapes = []
    for shape_filename in sorted(shapes_directory.glob("*.pkl")):
        shapes.append(pickle.load(open(shape_filename, "rb")))
    return shapes


def load_scene_element(optimization_directory, filename):
    filepath = Path(optimization_directory) / filename
    return pickle.load(open(filepath, "rb"))


def sample_dataset(true_images, shapes, num_samples, rng):
    num_samples = min(num_samples, len(true_images))
    sample_args = rng.choice(len(true_images), num_samples, replace=False)
    return true_images[sample_args], [shapes[arg] for arg in sample_args]


parser = argparse.ArgumentParser(description="Extract neural invariant maps")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--dataset_path", default="datasets", type=str)
parser.add_argument("--dataset_name", default="plain", type=str)
parser.add_argument("--parameters_wildcard", default="*SCENE-OPTIMIZATION")
parser.add_argument("--parameters_filename", default="parameters.json")
parser.add_argument("--shape_directory", default="optimized_shapes")
parser.add_argument("--scene_filename", default="scene.json")
parser.add_argument("--floor_filename", default="floor.npy")
parser.add_argument("--invariance_filename", default="invariances.json")
parser.add_argument("--label", default="INVARIANT-MAPS", type=str)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--num_images", default=50, type=int)
parser.add_argument("--num_samples", default=None, type=int)
parser.add_argument("--shadows", default=True, type=bool)
parser.add_argument("--model", default="VGG16", type=str)
parser.add_argument("--weights", default="imagenet", type=str)
parser.add_argument("--sparsity_threshold", default=0.5, type=float)
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)
RNG = np.random.default_rng(args.seed)

script_directory = Path(__file__).resolve().parent
root_path = resolve_path(args.root, script_directory)
dataset_root = resolve_path(args.dataset_path, script_directory)
paz.directory.make(str(dataset_root))
os.environ.setdefault("KERAS_HOME", str(dataset_root))

name = args.dataset_name
pattern = args.parameters_wildcard
name, wildcard = find_dataset_name(root_path, name, pattern)
args.dataset_name = name

label = args.label
model = args.model
root = setup_output_directory(root_path, name, model, label, args)

params_file = args.parameters_filename
metadata, opt_dir = load_optimization_metadata(wildcard, params_file)

factor = metadata["viewport_factor"]
params = compute_image_shape(name, factor)
image_shape, camera_origin, y_FOV = params

split = metadata["split"]
dataset = paz.datasets.fsclvr.load(name, split, image_shape)
true_images, depths, labels = paz.datasets.fsclvr.flatten(dataset)

shapes_dir = Path(opt_dir) / metadata["shapes_directory"]
shapes = load_shapes(shapes_dir)

if args.num_samples is not None:
    num_samples = args.num_samples
    true_images, shapes = sample_dataset(true_images, shapes, num_samples, RNG)

floor = load_scene_element(opt_dir, "floor.pkl")
lights = load_scene_element(opt_dir, "lights.pkl")

render = build_render(image_shape, y_FOV, camera_origin, args.shadows)
render = jax.jit(render)
pred_images = render_dataset(render, shapes, floor, lights)

weights = None if args.weights == "none" else args.weights
backbone = build_feature_backbone(model, weights)
model, preprocess, layer_names = backbone
layer_map = {f"layer_{i:02d}": n for i, n in enumerate(layer_names)}
layer_names_file = Path(root) / "layer_names.json"
paz.file.write_json(layer_map, str(layer_names_file))

image_args = RNG.integers(0, len(true_images), args.num_images)
layer_to_results, results = [], []
num_layers = len(layer_names)
for layer_arg, layer_name in enumerate(layer_names):
    layer_num = layer_arg + 1
    paz.message.info(f"Layer {layer_num} / {num_layers}: {layer_name}")
    feature_model = build_feature_model(model, layer_name)
    true_features = extract_features(true_images, feature_model, preprocess)
    pred_features = extract_features(pred_images, feature_model, preprocess)
    true_sparsity, true_magnitude = compute_sparsity(true_features)
    pred_sparsity, pred_magnitude = compute_sparsity(pred_features)
    large_true_sparsity = true_sparsity > args.sparsity_threshold
    large_pred_sparsity = pred_sparsity > args.sparsity_threshold
    if large_true_sparsity or large_pred_sparsity:
        paz.message.warn(
            f"{layer_name}: true_sparsity={true_sparsity:.1%}, "
            f"pred_sparsity={pred_sparsity:.1%}, "
            f"true_mag={true_magnitude:.4f}, pred_mag={pred_magnitude:.4f}"
        )
    dataset_error = compute_feature_error(true_features, pred_features)
    error_per_featuremap = np.mean(dataset_error, axis=0)
    invariant_indices = np.argsort(error_per_featuremap)[: args.top_k]
    invariant_errors = error_per_featuremap[invariant_indices]
    invariant_true_features = true_features[:, invariant_indices]
    invariant_pred_features = pred_features[:, invariant_indices]
    layer_path = Path(root) / f"layer_{layer_arg:02d}"
    paz.directory.make(str(layer_path))

    errors = invariant_errors.tolist()
    indices = invariant_indices.tolist()
    layer_results = {"errors": errors, "indices": indices}
    results_file = layer_path / "layer_results.json"
    paz.file.write_json(layer_results, str(results_file))
    results.extend([indices, errors])
    layer_to_results.append([indices, errors])

    for arg, image_arg in enumerate(image_args):
        image_path = layer_path / f"{arg:02d}_images"
        paz.directory.make(str(image_path))
        true_image = true_images[image_arg]
        pred_image = pred_images[image_arg]
        write_image(true_image, image_path, "true_image.png")
        write_image(pred_image, image_path, "pred_image.png")

        true_featuremaps = invariant_true_features[image_arg]
        pred_featuremaps = invariant_pred_features[image_arg]
        true_featuremaps = process_features(true_featuremaps)
        pred_featuremaps = process_features(pred_featuremaps)
        iterator = enumerate(zip(true_featuremaps, pred_featuremaps))
        for feature_arg, (true_featuremap, pred_featuremap) in iterator:
            true_filename = f"{feature_arg:02d}_true_featuremap.png"
            pred_filename = f"{feature_arg:02d}_pred_featuremap.png"
            write_featuremap(true_featuremap, image_path, true_filename)
            write_featuremap(pred_featuremap, image_path, pred_filename)


invariances = extract_invariances(layer_to_results, layer_names, args.top_k)
invariances_file = Path(root) / args.invariance_filename
paz.file.write_json(invariances, str(invariances_file))
layer_labels, label_names = [], []
for layer_arg in range(len(layer_names)):
    layer_label = f"layer_{layer_arg:02d}"
    layer_labels.extend([layer_label, layer_label])
    label_names.extend(["featuremap", "error"])
column_names = zip(layer_labels, label_names)
columns = pd.MultiIndex.from_tuples(column_names)
results = np.array(results)
results = np.moveaxis(results, 0, 1)
index = list(range(args.top_k))
data_frame = pd.DataFrame(results, index=index, columns=columns)
for name in layer_labels:
    col = data_frame[name, "featuremap"]
    data_frame[name, "featuremap"] = col.astype(int)
data_frame.style.to_latex(str(Path(root) / "results.tex"))
data_frame.to_csv(str(Path(root) / "results.csv"))
