import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import time
import json
import pickle
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("KERAS_BACKEND", "jax")

import arviz as az
import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_probability.substrates import jax as tfp

import paz
from observation_model import (
    build_observation_model,
    build_render_function,
    parse_summary,
)

tfd = tfp.distributions


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


def mode(values):
    return az.plots.plot_utils.calculate_point_estimate("mode", values)


def median(values):
    return az.plots.plot_utils.calculate_point_estimate("median", values)


def build_summary(trace):
    variable_names = list(trace.keys())
    return az.summary(
        trace, var_names=variable_names, stat_funcs=[mode, median]
    )


def write_image(image, directory, filename):
    filepath = Path(directory) / filename
    image = paz.image.denormalize(jp.clip(image, 0.0, 1.0))
    paz.image.write(filepath, image)


def write_true_pred_image(true_image, pred_image, directory, label):
    image = jp.concatenate([true_image, pred_image], axis=1)
    write_image(image, directory, f"true_pred_{label}_image.png")


def write_error_image(true_image, pred_image, directory):
    image = (true_image - pred_image) ** 2
    write_image(image, directory, "error_image.png")


def write_trace(trace, directory):
    paz.pytree.to_pickle(trace, Path(directory) / "trace.pkl")


def write_summary(summary, directory):
    summary.to_csv(Path(directory) / "summary.csv")
    summary.to_latex(Path(directory) / "summary.tex")


def plot_trace(inference_data, directory):
    az.plot_trace(inference_data)
    plt.savefig(Path(directory) / "trace.pdf", bbox_inches="tight")
    plt.close()


def plot_shift_posterior(trace, label, directory, name="shift"):
    x_samples = trace[name][:, :, 0]
    y_samples = trace[name][:, :, 1]
    xz_trace = {"x": x_samples, "y": y_samples}
    axes = az.plot_pair(xz_trace, var_names=["x", "y"])
    label = list(label.values())[0]
    true_shift = paz.datasets.fsclvr.parse_shift(label)
    if hasattr(axes, "shape"):
        axis = axes[0, 0]
    else:
        axis = axes
    axis.scatter(true_shift[0], true_shift[1], s=30, marker="*", c="red")
    plt.savefig(Path(directory) / f"{name}_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_theta_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["theta"])
    plt.savefig(Path(directory) / "theta_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_trace_variable(trace, name, directory):
    az.plot_trace(trace, var_names=[name])
    plt.savefig(Path(directory) / f"{name}_trace.pdf", bbox_inches="tight")
    plt.close()


def plot_shift_posteriors(trace, directory):
    az.plot_posterior(trace, var_names=["shift"])
    plt.savefig(Path(directory) / "shift_posteriors.pdf", bbox_inches="tight")
    plt.close()


def plot_shift_x_posterior(trace, directory):
    data = {"shift_x": trace["shift"][:, :, 0]}
    az.plot_posterior(data, var_names=["shift_x"])
    plt.savefig(Path(directory) / "shift_x_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_shift_y_posterior(trace, directory):
    data = {"shift_y": trace["shift"][:, :, 1]}
    az.plot_posterior(data, var_names=["shift_y"])
    plt.savefig(Path(directory) / "shift_y_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_scale_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["scale"])
    plt.savefig(Path(directory) / "scale_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_scale_x_posterior(trace, directory):
    data = {"scale_x": trace["scale"][:, :, 0]}
    az.plot_posterior(data, var_names=["scale_x"])
    plt.savefig(Path(directory) / "scale_x_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_scale_y_posterior(trace, directory):
    data = {"scale_y": trace["scale"][:, :, 1]}
    az.plot_posterior(data, var_names=["scale_y"])
    plt.savefig(Path(directory) / "scale_y_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_scale_z_posterior(trace, directory):
    data = {"scale_z": trace["scale"][:, :, 2]}
    az.plot_posterior(data, var_names=["scale_z"])
    plt.savefig(Path(directory) / "scale_z_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_color_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["color"])
    plt.savefig(Path(directory) / "color_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_ambient_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["ambient"])
    plt.savefig(Path(directory) / "ambient_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_diffuse_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["diffuse"])
    plt.savefig(Path(directory) / "diffuse_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_specular_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["specular"])
    plt.savefig(Path(directory) / "specular_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_shininess_posterior(trace, directory):
    az.plot_posterior(trace, var_names=["shininess"])
    plt.savefig(
        Path(directory) / "shininess_posterior.pdf", bbox_inches="tight"
    )
    plt.close()


def plot_classes_posterior(trace, directory):
    classes = trace["classes"]
    data = {
        "class_0": classes[:, :, 0],
        "class_1": classes[:, :, 1],
        "class_2": classes[:, :, 2],
    }
    az.plot_posterior(data)
    plt.savefig(Path(directory) / "classes_posterior.pdf", bbox_inches="tight")
    plt.close()


def plot_dirichlet_posterior(trace, directory):
    classes = trace["classes"]
    data = {
        "class_0": classes[:, :, 0],
        "class_1": classes[:, :, 1],
        "class_2": classes[:, :, 2],
    }
    az.plot_posterior(data)
    plt.savefig(
        Path(directory) / "dirichlet_posterior.pdf", bbox_inches="tight"
    )
    plt.close()


def estimate_point(summary, render, statistic):
    sample = parse_summary(summary, statistic)
    return render(sample)


def iterate_samples(samples, num_samples):
    if num_samples == 1:
        yield samples._asdict()
        return
    sample_dict = samples._asdict()
    for index in range(num_samples):
        yield {name: value[index] for name, value in sample_dict.items()}


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
parser.add_argument("--num_posterior_samples", default=100, type=int)
parser.add_argument("--image_noise", default=1.0, type=float)
parser.add_argument("--neuro_weight", default=0.05, type=float)
parser.add_argument("--num_branches", default=3, type=int)
parser.add_argument("--num_chains", default=20, type=int)
parser.add_argument("--burn_in", default=1000, type=int)
parser.add_argument("--num_samples", default=30_000, type=int)
parser.add_argument("--sigma", default=0.05, type=float)
parser.add_argument("--tune", default=True, type=bool)
parser.add_argument("--progress", default=True, type=bool)
args = parser.parse_args()

jax.config.update("jax_platform_name", args.device)
keras.utils.set_random_seed(args.seed)

key = jax.random.PRNGKey(args.seed)
key_0, key_1, key_2 = jax.random.split(key, 3)

root_path = Path(args.root) / args.dataset_name
scene_pattern = root_path / args.scene_wildcard
priors_pattern = root_path / args.priors_wildcard
features_pattern = root_path / args.features_wildcard

scene_metadata, scene_directory = load_latest_parameters(
    scene_pattern, args.params_filename
)
dataset_name = scene_metadata["dataset_name"]
viewport_factor = scene_metadata["viewport_factor"]
image_shape, camera_origin, y_FOV = (
    None,
    None,
    None,
)
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
render = build_render_function(
    image_shape, y_FOV, camera_pose, lights, args.shadows
)
observation_model = build_observation_model(render, floor)

invariances = paz.file.load_latest(features_pattern, args.features_filename)
neuro_model = None
if args.neuro_weight > 0.0 and args.num_branches > 0:
    weights = None if args.weights == "none" else args.weights
    model_args = args.model, weights, invariances, args.num_branches
    branch_model, preprocess = build_branch_model(*model_args)
    neuro_model = build_neuro_likelihood(
        args.neuro_weight, branch_model, preprocess
    )

noise_model = tfd.TruncatedNormal(0.0, args.image_noise, -1.0, 1.0)
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
paz.file.write_json(args.__dict__, Path(root) / "parameters.json")
paz.file.write_json(results, Path(root) / "results.json")

trace_dict = build_trace(posterior.samples, args.burn_in)
inference_data = az.from_dict(posterior=trace_dict)
summary = build_summary(trace_dict)
write_summary(summary, root)
write_trace(trace_dict, root)

write_image(image, root, "true_image.png")
mean_image, _ = estimate_point(summary, observation_model, "mean")
medn_image, _ = estimate_point(summary, observation_model, "median")
write_image(mean_image, root, "mean_point.png")
write_image(medn_image, root, "median_point.png")
write_true_pred_image(image, mean_image, root, "mean")
write_true_pred_image(image, medn_image, root, "median")
write_error_image(image, mean_image, root)
write_error_image(image, medn_image, root)
write_image(mean_image, root, "mean_image.png")
write_image(medn_image, root, "median_image.png")

plot_trace(inference_data, root)
plot_shift_posterior(trace_dict, label, root)
plot_theta_posterior(inference_data, root)
plot_shift_posteriors(inference_data, root)
plot_shift_x_posterior(trace_dict, root)
plot_shift_y_posterior(trace_dict, root)
plot_scale_posterior(inference_data, root)
plot_scale_x_posterior(trace_dict, root)
plot_scale_y_posterior(trace_dict, root)
plot_scale_z_posterior(trace_dict, root)
plot_color_posterior(inference_data, root)
plot_ambient_posterior(inference_data, root)
plot_diffuse_posterior(inference_data, root)
plot_specular_posterior(inference_data, root)
plot_shininess_posterior(inference_data, root)
plot_classes_posterior(trace_dict, root)
plot_dirichlet_posterior(trace_dict, root)

posterior_directory = Path(root) / "posterior_samples"
paz.directory.make(posterior_directory)
forward_samples = posterior.sample(key_1, args.num_posterior_samples)
iterator = iterate_samples(forward_samples, args.num_posterior_samples)
for arg, sample in enumerate(iterator):
    pred_image, _ = observation_model(sample)
    filename = f"posterior_sample_{arg:02d}.png"
    write_image(pred_image, posterior_directory, filename)
