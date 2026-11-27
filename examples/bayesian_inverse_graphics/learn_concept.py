import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import time
import json
import pickle
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jp
import numpy as np
import pandas
import keras

import paz
import figures
import paz.utils.pytree as pytree
from observation_model import build_observation_model, build_render_function


NAME_ORDER = [
    "shift",
    "theta",
    "scale",
    "color",
    "ambient",
    "diffuse",
    "specular",
    "shininess",
    "classes",
]


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


def build_branches(invariances, num_branches):
    items = sorted(invariances.items(), key=lambda item: int(item[0]))
    branches = [item[1] for item in items]
    return branches[:num_branches]


def build_branch_model(model_name, weights, invariances, num_branches):
    model, preprocess, _ = build_feature_backbone(model_name, weights)
    branches = build_branches(invariances, num_branches)
    layer_names = {branch["layer"] for branch in branches}
    layer_models = {
        name: build_feature_model(model, name) for name in layer_names
    }

    def extract_featuremap(layer_output, feature_arg):
        featuremap = layer_output[0]
        if featuremap.ndim == 3:
            featuremap = jp.transpose(featuremap, (2, 0, 1))
        return featuremap[int(feature_arg)]

    def apply(image):
        image = jp.expand_dims(image, axis=0)
        layer_outputs = {
            name: model(image, training=False)
            for name, model in layer_models.items()
        }
        features = []
        for branch in branches:
            layer_name = branch["layer"]
            feature_arg = branch["featuremap"]
            featuremap = extract_featuremap(
                layer_outputs[layer_name], feature_arg
            )
            features.append(featuremap)
        return features

    return apply, preprocess


def preprocess_input(image, preprocess):
    image = image * 255.0
    return preprocess(image)


def compute_feature_loss(true_features, pred_features):
    losses = []
    for true_feature, pred_feature in zip(true_features, pred_features):
        loss = (true_feature - pred_feature) ** 2
        loss = jp.mean(loss, axis=(0, 1))
        losses.append(loss)
    return jp.array(losses)


def build_neuro_likelihood(weight, branch_model, preprocess):
    def apply(true_image, pred_image):
        true_image = preprocess_input(true_image, preprocess)
        pred_image = preprocess_input(pred_image, preprocess)
        true_features = branch_model(true_image)
        pred_features = branch_model(pred_image)
        losses = compute_feature_loss(true_features, pred_features)
        return -weight * losses.sum()

    return apply


class ImageLikelihood:
    def __init__(self, sample, observation_model, noise_model, neuro_model):
        self.forward_sample = sample
        self.observation_model = observation_model
        self.noise_model = noise_model
        self.neuro_model = neuro_model

    def log_prob(self, true_image):
        pred_image, _ = self.observation_model(self.forward_sample)
        color_log_prob = self.noise_model.log_prob(
            true_image - pred_image
        ).sum()
        if self.neuro_model is None:
            return color_log_prob
        return color_log_prob + self.neuro_model(true_image, pred_image)

    def sample(self, num_samples, seed=None):
        pred_image, _ = self.observation_model(self.forward_sample)
        pred_batch = jp.broadcast_to(
            pred_image, (num_samples,) + pred_image.shape
        )
        noise = self.noise_model.sample(num_samples, seed=seed)
        return pred_batch + noise


def build_image_distribution(observation_model, noise_model, neuro_model):
    def distribution_fn(
        shift,
        theta,
        scale,
        color,
        ambient,
        diffuse,
        specular,
        shininess,
        classes,
    ):
        sample = {
            "shift": shift,
            "theta": theta,
            "scale": scale,
            "color": color,
            "ambient": ambient,
            "diffuse": diffuse,
            "specular": specular,
            "shininess": shininess,
            "classes": classes,
        }
        return ImageLikelihood(
            sample, observation_model, noise_model, neuro_model
        )

    return distribution_fn


def load_json(path):
    with open(path, "r") as filedata:
        return json.load(filedata)


def load_latest_parameters(wildcard, filename):
    directory = paz.directory.find_latest(str(wildcard))
    return load_json(Path(directory) / filename), Path(directory)


def load_latest_priors(wildcard, prior_filename):
    directory = paz.directory.find_latest(str(wildcard))
    path = Path(directory) / prior_filename
    return paz.inference.load(path).inputs, Path(directory)


def load_scene_elements(scene_directory):
    floor = pickle.load(open(scene_directory / "floor.pkl", "rb"))
    lights = pickle.load(open(scene_directory / "lights.pkl", "rb"))
    return floor, lights


def ensure_prior_order(priors):
    name_to_prior = {prior.name: prior for prior in priors}
    missing = [name for name in NAME_ORDER if name not in name_to_prior]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing priors: {missing_list}")
    return [name_to_prior[name] for name in NAME_ORDER]


def build_trace(samples, burn_in):
    trace = {}
    sample_dict = samples._asdict()
    for name, values in sample_dict.items():
        values = values[burn_in:]
        values = jp.swapaxes(values, 0, 1)
        trace[name] = np.asarray(values)
    return trace


def build_summary(trace):
    rows = {}
    for name, values in trace.items():
        values = values.reshape(values.shape[0] * values.shape[1], -1)
        for arg, samples in enumerate(values.T):
            is_scalar = values.shape[1] == 1
            row_name = name if is_scalar else f"{name}[{arg}]"
            rows[row_name] = summarize(samples)
    return pandas.DataFrame.from_dict(rows, orient="index")


def summarize(samples):
    return {
        "mean": samples.mean(),
        "stdv": samples.std(),
        "median": np.median(samples),
        "mode": compute_mode(samples),
    }


def compute_mode(samples):
    # A density cannot be estimated from samples that are all equal.
    if np.ptp(samples) > 0.0:
        grid, density = figures.compute_density(samples)
        mode = grid[np.argmax(density)]
    else:
        mode = samples[0]
    return mode


def write_image(image, directory, filename):
    filepath = Path(directory) / filename
    image = paz.image.denormalize(jp.clip(image, 0.0, 1.0))
    paz.image.write(filepath, image)


def write_trace(trace, directory):
    pytree.to_pickle(trace, Path(directory) / "trace.pkl")


def write_summary(summary, directory):
    summary.to_csv(Path(directory) / "summary.csv")
    summary.to_latex(Path(directory) / "summary.tex")


def build_output_directory(root, dataset_name, label, concept, shot_arg):
    root = paz.directory.make_timestamped(Path(root) / dataset_name, label)
    root = Path(root) / f"CONCEPT-{concept:02d}" / f"SHOT-{shot_arg:02d}"
    paz.directory.make(root)
    return root


parser = argparse.ArgumentParser(
    description="Probabilistic inverse graphics for few-shot learning"
)
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default="IMAGE", type=str)
parser.add_argument("--split", default="test", type=str)
parser.add_argument("--device", default="gpu", type=str)
parser.add_argument("--dataset_name", default="plain", type=str)
parser.add_argument("--prior_filename", default="priors", type=str)
parser.add_argument("--priors_wildcard", default="*PRIORS", type=str)
parser.add_argument("--params_filename", default="parameters.json", type=str)
parser.add_argument("--scene_wildcard", default="*SCENE-OPTIMIZATION", type=str)
parser.add_argument("--features_wildcard", default="*INVARIANT-MAPS", type=str)
parser.add_argument("--features_filename", default="invariances.json", type=str)
parser.add_argument("--shadows", default=False, type=bool)
parser.add_argument("--model", default="VGG16", type=str)
parser.add_argument("--weights", default="imagenet", type=str)
parser.add_argument("--concept", default=0, type=int)
parser.add_argument("--shot_arg", default=0, type=int)
parser.add_argument("--tune_steps", default=500, type=int)
parser.add_argument("--tune_episodes", default=5, type=int)
parser.add_argument("--image_noise", default=1.0, type=float)
parser.add_argument("--neuro_weight", default=0.05, type=float)
parser.add_argument("--num_branches", default=3, type=int)
parser.add_argument("--num_chains", default=20, type=int)
parser.add_argument("--burn_in", default=3000, type=int)
parser.add_argument("--num_samples", default=30_000, type=int)
parser.add_argument("--sigma", default=0.05, type=float)
parser.add_argument("--tune", default=True, type=bool)
parser.add_argument("--progress", default=True, type=bool)
args = parser.parse_args()

jax.config.update("jax_platform_name", args.device)
keras.utils.set_random_seed(args.seed)

key = jax.random.PRNGKey(args.seed)
# Three keys were split here before; keeping it preserves the chains.
key_0, _, _ = jax.random.split(key, 3)

root_path = Path(args.root) / args.dataset_name
scene_pattern = root_path / args.scene_wildcard
priors_pattern = root_path / args.priors_wildcard
features_pattern = root_path / args.features_wildcard

scene_metadata, scene_directory = load_latest_parameters(
    scene_pattern, args.params_filename
)
dataset_name = scene_metadata["dataset_name"]
viewport_factor = scene_metadata["viewport_factor"]
dataset_metadata = paz.datasets.fsclvr.parse_metadata(dataset_name)
camera_origin = jp.array(dataset_metadata["camera_origin"])
H, W = dataset_metadata["image_shape"]
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
y_FOV = dataset_metadata["y_FOV"]

priors, priors_directory = load_latest_priors(
    priors_pattern, args.prior_filename
)
priors = ensure_prior_order(priors)

floor, lights = load_scene_elements(scene_directory)
camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = paz.SE3.view_transform(
    camera_origin, camera_target, camera_upward
)
intrinsics = paz.graphics.camera.compute_intrinsics(y_FOV, *image_shape)
render = build_render_function(
    image_shape, intrinsics, camera_pose, lights, args.shadows
)
observation_model = build_observation_model(render, floor)

features_directory = paz.directory.find_latest(str(features_pattern))
invariances = load_json(Path(features_directory) / args.features_filename)
neuro_model = None
if args.neuro_weight > 0.0 and args.num_branches > 0:
    weights = None if args.weights == "none" else args.weights
    model_args = args.model, weights, invariances, args.num_branches
    branch_model, preprocess = build_branch_model(*model_args)
    neuro_model = build_neuro_likelihood(
        args.neuro_weight, branch_model, preprocess
    )

noise_args = (0.0, args.image_noise, -1.0, 1.0)
noise_model = paz.distributions.TruncatedNormal(*noise_args)
distribution_fn = build_image_distribution(
    observation_model, noise_model, neuro_model
)
image_node = paz.Observable(distribution_fn, name="image")(*priors)
model = paz.PGM(priors, [image_node], "learn_concept")
tuner = paz.AdaptiveStepTuner(
    sigma=args.sigma,
    num_steps=args.tune_steps,
    num_episodes=args.tune_episodes,
    progress=args.progress,
)
model.configure(num_chains=args.num_chains, tuner=tuner, warmup=0)

dataset = paz.datasets.fsclvr.load(dataset_name, args.split, image_shape)
image, depth, label = dataset[args.concept][args.shot_arg]
data = {"image": image}

start_time = time.perf_counter()
posterior = model.infer(
    key_0,
    data,
    num_samples=args.num_samples,
    tune=args.tune,
    sigma=args.sigma,
    progress=args.progress,
)
end_time = time.perf_counter()

diagnostics = posterior.diagnostics()
results = {
    "total_duration": end_time - start_time,
    "acceptance_rate": diagnostics["acceptance_rate"].tolist(),
    "mean_acceptance_rate": float(diagnostics["mean_acceptance_rate"]),
    "sigma": float(posterior.config["sigma"]),
}

root = build_output_directory(
    args.root, dataset_name, args.label, args.concept, args.shot_arg
)
# Later renders and redraws need the exact inputs this run was fitted with.
input_directories = (scene_directory, priors_directory, features_directory)
input_names = ("scene_directory", "priors_directory", "features_directory")
resolved = [str(Path(path).resolve()) for path in input_directories]
inputs = dict(zip(input_names, resolved))
paz.file.write_json({**args.__dict__, **inputs}, Path(root) / "parameters.json")
paz.file.write_json(results, Path(root) / "results.json")

trace_dict = build_trace(posterior.samples, args.burn_in)
write_trace(trace_dict, root)
write_image(image, root, "true_image.png")
paz.file.write_json(list(label.values())[0], Path(root) / "label.json")
write_summary(build_summary(trace_dict), root)
