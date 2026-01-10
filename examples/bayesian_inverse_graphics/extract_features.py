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
    camera_target = jp.array([0.0, 0.0, 0.0])
    camera_upward = jp.array([0.0, 1.0, 0.0])
    camera_pose = paz.SE3.view_transform(
        jp.array(camera_origin), camera_target, camera_upward
    )
    rays = paz.graphics.camera.build_rays(image_shape, y_FOV, camera_pose)
    return paz.partial(
        paz.graphics.render,
        image_shape=image_shape,
        world_to_camera=camera_pose,
        rays=rays,
        shadows=shadows,
    )


def render_dataset(render, shapes, floor, lights):
    images = []
    for shape in shapes:
        scene = paz.graphics.Scene([shape, floor])
        image, depth = render(scene=scene, mask=None, lights=lights)
        images.append(jp.clip(image, 0.0, 1.0))
    return np.array(images)


def build_feature_backbone(model_name, weights):
    if model_name == "VGG16":
        model = keras.applications.VGG16(include_top=False, weights=weights)
        preprocess = keras.applications.vgg16.preprocess_input
        layer_names = [
            "block1_conv1",
            "block1_conv2",
            "block2_conv1",
            "block2_conv2",
            "block3_conv1",
            "block3_conv2",
            "block3_conv3",
            "block4_conv1",
            "block4_conv2",
            "block4_conv3",
            "block5_conv1",
            "block5_conv2",
            "block5_conv3",
        ]
        return model, preprocess, layer_names

    if model_name == "TINYCONVNEXT":
        model = keras.applications.convnext.ConvNeXtTiny(
            include_top=False, weights=weights
        )
        preprocess = keras.applications.convnext.preprocess_input
        layer_names = [
            "stem",
            "stage_0_block_0",
            "stage_1_block_0",
            "stage_2_block_0",
            "stage_3_block_0",
        ]
        return model, preprocess, layer_names

    raise ValueError(f"Unknown model '{model_name}'")


def load_model_weights(model, weights_path):
    if weights_path is None:
        return model
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model.load_weights(str(weights_path))
    return model


def build_feature_model(model, layer_name):
    return keras.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )


def extract_image_features(image, model, preprocess):
    image = preprocess(image)
    image = jp.expand_dims(image, axis=0)
    featuremap = model(image, training=False)
    featuremap = np.array(featuremap)[0]
    if featuremap.ndim == 3:
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


def extract_invariances(layer_to_results, layer_names, top_k=5):
    layer_to_errors, layer_to_features = [], []
    for layer in layer_to_results:
        featuremap, errors = layer
        layer_to_errors.append(errors)
        layer_to_features.append(featuremap)
    layer_to_errors = np.array(layer_to_errors)
    errors = layer_to_errors.reshape(-1)
    num_features = layer_to_errors.shape[1]

    lower_error_args = np.argsort(errors)
    row_args = lower_error_args // num_features
    col_args = lower_error_args - (row_args * num_features)

    row_args = row_args[:top_k]
    col_args = col_args[:top_k]

    invariances = {}
    for arg, (row_arg, col_arg) in enumerate(zip(row_args, col_args)):
        model_layer = layer_names[row_arg]
        feature_arg = int(layer_to_features[row_arg][col_arg])
        invariances[arg] = {"layer": model_layer, "featuremap": feature_arg}
    return invariances


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


parser = argparse.ArgumentParser(description="Extract neural invariant maps")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--dataset_path", default="datasets", type=str)
parser.add_argument("--dataset_name", default="PRIMITIVES", type=str)
parser.add_argument("--parameters_wildcard", default="*SCENE-OPTIMIZATION")
parser.add_argument("--parameters_filename", default="parameters.json")
parser.add_argument("--shape_directory", default="optimized_shapes")
parser.add_argument("--scene_filename", default="scene.json")
parser.add_argument("--floor_filename", default="floor.npy")
parser.add_argument("--invariance_filename", default="invariances.json")
parser.add_argument("--weights_path", default=None)
parser.add_argument("--label", default="INVARIANT-MAPS", type=str)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--num_images", default=50, type=int)
parser.add_argument("--num_samples", default=None, type=int)
parser.add_argument("--shadows", default=True, type=bool)
parser.add_argument(
    "--model", default="VGG16", choices=["VGG16", "TINYCONVNEXT"], type=str
)
parser.add_argument(
    "--weights", default="none", choices=["none", "imagenet"], type=str
)
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)
RNG = np.random.default_rng(args.seed)

script_directory = Path(__file__).resolve().parent
root_path = Path(args.root)
if not root_path.is_absolute():
    root_path = script_directory / root_path

dataset_root = Path(args.dataset_path)
if not dataset_root.is_absolute():
    dataset_root = script_directory / dataset_root
paz.directory.make(str(dataset_root))
os.environ.setdefault("KERAS_HOME", str(dataset_root))

wildcard = str(root_path / args.dataset_name / args.parameters_wildcard)
if len(glob(wildcard)) == 0:
    dataset_candidates = [
        path for path in root_path.iterdir() if path.is_dir()
    ]
    if len(dataset_candidates) == 1:
        args.dataset_name = dataset_candidates[0].name
        wildcard = str(
            root_path / args.dataset_name / args.parameters_wildcard
        )

label = "-".join([args.model, args.label])
root = paz.directory.make_timestamped(
    str(root_path / args.dataset_name), label
)
paz.file.write_json(args.__dict__, str(Path(root) / args.parameters_filename))
optimization_metadata = paz.file.load_latest(wildcard, args.parameters_filename)
viewport_factor = optimization_metadata["viewport_factor"]
split = optimization_metadata["split"]
dataset_metadata = paz.datasets.fsclvr.parse_metadata(args.dataset_name)
H, W = dataset_metadata["image_shape"]
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
camera_origin = jp.array(dataset_metadata["camera_origin"])
y_FOV = dataset_metadata["y_FOV"]

dataset = paz.datasets.fsclvr.load(args.dataset_name, split, image_shape)
true_images, depths, labels = paz.datasets.fsclvr.flatten(dataset)

optimization_directory = paz.directory.find_latest(wildcard)
shapes_directory = (
    Path(optimization_directory) / optimization_metadata["shapes_directory"]
)
shapes = []
for shape_filename in sorted(shapes_directory.glob("*.pkl")):
    shapes.append(pickle.load(open(shape_filename, "rb")))

if args.num_samples is not None:
    num_samples = min(args.num_samples, len(true_images))
    sample_args = RNG.choice(len(true_images), num_samples, replace=False)
    true_images = true_images[sample_args]
    shapes = [shapes[arg] for arg in sample_args]

floor_filename = Path(optimization_directory) / "floor.pkl"
floor = pickle.load(open(floor_filename, "rb"))

lights_filename = Path(optimization_directory) / "lights.pkl"
lights = pickle.load(open(lights_filename, "rb"))

render = build_render(image_shape, y_FOV, camera_origin, args.shadows)
render = jax.jit(render)
pred_images = render_dataset(render, shapes, floor, lights)

weights = None if args.weights == "none" else args.weights
model, preprocess, layer_names = build_feature_backbone(args.model, weights)
model = load_model_weights(model, args.weights_path)
layer_mapping = {
    f"layer_{layer_arg:02d}": layer_name
    for layer_arg, layer_name in enumerate(layer_names)
}
paz.file.write_json(layer_mapping, str(Path(root) / "layer_names.json"))

image_args = RNG.integers(0, len(true_images), args.num_images)
layer_to_results, results = [], []
num_layers = len(layer_names)
for layer_arg, layer_name in enumerate(layer_names):
    paz.message.info(
        f"Processing layer {layer_arg + 1} / {num_layers}: {layer_name}"
    )
    feature_model = build_feature_model(model, layer_name)
    true_features = extract_features(true_images, feature_model, preprocess)
    pred_features = extract_features(pred_images, feature_model, preprocess)
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
    paz.file.write_json(layer_results, str(layer_path / "layer_results.json"))
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
paz.file.write_json(invariances, str(Path(root) / args.invariance_filename))
layer_labels = []
label_names = []
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
    data_frame[name, "featuremap"] = data_frame[name, "featuremap"].astype(int)
data_frame.style.to_latex(str(Path(root) / "results.tex"))
data_frame.to_csv(str(Path(root) / "results.csv"))
