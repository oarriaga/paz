import argparse
import glob
import json
import pickle
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.utils.plot as plot
from paz.backend import directory
from paz.inference.gmm.model import GMM
from observation_model import build_observation_model, build_render_function

tfd = tfp.distributions
tfb = tfp.bijectors

parser = argparse.ArgumentParser(description="Optimize Bijectors")
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--label", default="PRIORS", type=str)
parser.add_argument("--dataset_name", default="plain", type=str)
parser.add_argument("--parameters_wildcard", default="*SCENE-OPTIMIZATION")
parser.add_argument("--parameters_filename", default="parameters.json")
parser.add_argument("--scene_filename", default="scene.json")
parser.add_argument("--materials_filename", default="materials.json")
parser.add_argument("--floor_filename", default="floor.npy")
parser.add_argument("--prior_filename", default="priors")
parser.add_argument("--num_samples", default=100_000, type=int)
parser.add_argument("--gradient_steps", default=25_000, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=int)
parser.add_argument("--shift_mean", nargs="+", default=[0.0, 0.025])
parser.add_argument("--shift_scale", nargs="+", default=[0.08, 0.08])
parser.add_argument("--theta_mean", default=0.0, type=float)
parser.add_argument("--theta_concentration", default=0.0, type=float)
parser.add_argument("--scale_mean", nargs="+", default=[0.025, 0.025, 0.025])
parser.add_argument("--scale_scale", nargs="+", default=[1e-4, 1e-4, 1e-4])
parser.add_argument("--shadows", default=True, type=bool)
parser.add_argument("--num_images", default=100, type=int)
parser.add_argument("--classes_temperature", default=0.5, type=float)
parser.add_argument(
    "--class_probabilities", nargs="+", default=[1 / 3, 1 / 3, 1 / 3]
)
args = parser.parse_args()


BijectorParams = namedtuple("BijectorParams", ["shift", "scale"])


NAME_TO_TEX = {
    "shift": r"$x$-$y$ translation [m]",
    "x-translation": r"$x$ [m]",
    "y-translation": r"$y$ [m]",
    "theta": r"$\theta$ [rad]",
    "scale": r"scale",
    "x-scale": r"scale $s_x$ [m]",
    "y-scale": r"scale $s_y$ [m]",
    "z-scale": r"scale $s_z$ [m]",
    "color": r"color [rgb]",
    "ambient": r"ambient [$k_a$]",
    "diffuse": r"diffuse [$k_d$]",
    "specular": r"specular [$k_s$]",
    "shininess": r"shininess [$\alpha$]",
    "classes": r"classes",
    "R-color": r"color [r]",
    "G-color": r"color [g]",
    "B-color": r"color [b]",
}


def load_parameters(wildcard, filename):
    filepath = Path(directory.find_latest(wildcard)) / filename
    if filepath.exists():
        with filepath.open("r") as filedata:
            return json.load(filedata)
    pickle_path = filepath.with_suffix(".pkl")
    if pickle_path.exists():
        with pickle_path.open("rb") as filedata:
            return pickle.load(filedata)
    message = f"Could not find {filename} or {pickle_path.name}"
    raise FileNotFoundError(message)


def build_directory(root, label):
    return Path(directory.make_timestamped(root, label))


def write_image(image, output_directory, filename):
    if not filename.endswith(".png"):
        raise ValueError(f"Filename {filename} missing extension .png")
    filepath = Path(output_directory) / filename
    plt.imsave(filepath, image)


def write_losses(losses, output_directory, filename):
    figure, axis = plot.subplots()
    steps = np.arange(len(losses))
    plot.line(steps, losses, axis=axis, color=plot.DANDELION.primary)
    plot.set_labels(axis, x="step", y="loss")
    plot.hide_spines(axis, "box")
    fullpath = Path(output_directory) / filename
    if fullpath.suffix != ".pdf":
        raise ValueError(f"Filename {fullpath} missing extension .pdf")
    plot.save(figure, fullpath)


def plot_forward_samples(forward_samples, name, output_directory):
    figure, axis = plot.subplots()
    finite_samples = forward_samples[np.isfinite(forward_samples)]
    plot.density(
        finite_samples,
        axis=axis,
        color=plot.DANDELION.primary,
        alpha=0.4,
        linewidth=5.0,
    )
    plot.clean(axis)
    plot.set_labels(
        axis, x=NAME_TO_TEX.get(name, name), y="density", labelpad=20
    )
    filepath = Path(output_directory) / f"prior_forward_{name}.pdf"
    plot.save(figure, filepath)


def plot_inverse_samples(
    inverse_samples, normals_samples, name, output_directory
):
    figure, axis = plot.subplots()
    finite_inverse = inverse_samples[np.isfinite(inverse_samples)]
    finite_normals = normals_samples[np.isfinite(normals_samples)]
    plot.density(
        finite_normals,
        axis=axis,
        color="tab:red",
        alpha=0.4,
        linewidth=5.0,
        label="normal",
    )
    plot.density(
        finite_inverse,
        axis=axis,
        color="tab:blue",
        alpha=0.4,
        linewidth=5.0,
        label="inverse",
    )
    plot.clean(axis)
    plot.legend(axis, frameon=False, fontsize=10)
    plot.set_labels(
        axis, x=NAME_TO_TEX.get(name, name), y="density", labelpad=20
    )
    filepath = Path(output_directory) / f"prior_inverse_{name}.pdf"
    plot.save(figure, filepath)


def plot_prior(key, prior_node, output_directory, num_samples=10_000):
    key_0, key_1, key_2 = jax.random.split(key, 3)
    forward_samples = prior_node.sample(key_0, num_samples)
    inverse_samples = prior_node.sample_inverse(key_1, num_samples)
    normals_samples = tfd.Normal(0.0, 1.0).sample(num_samples, seed=key_2)
    name = prior_node.name

    if name == "shift":
        for arg, shift_name in enumerate(["x-translation", "y-translation"]):
            plot_forward_samples(
                forward_samples[:, arg], shift_name, output_directory
            )
            plot_inverse_samples(
                inverse_samples[:, arg],
                normals_samples,
                shift_name,
                output_directory,
            )
    elif name == "color":
        for arg, channel_name in enumerate(["R-color", "G-color", "B-color"]):
            plot_forward_samples(
                forward_samples[:, arg], channel_name, output_directory
            )
            plot_inverse_samples(
                inverse_samples[:, arg],
                normals_samples,
                channel_name,
                output_directory,
            )
    elif name == "scale":
        for arg, scale_name in enumerate(["x-scale", "y-scale", "z-scale"]):
            plot_forward_samples(
                forward_samples[:, arg], scale_name, output_directory
            )
            plot_inverse_samples(
                inverse_samples[:, arg],
                normals_samples,
                scale_name,
                output_directory,
            )
    else:
        plot_forward_samples(forward_samples, name, output_directory)
        plot_inverse_samples(
            inverse_samples, normals_samples, name, output_directory
        )


def plot_priors(key, priors, output_directory, num_samples=10_000):
    keys = jax.random.split(key, len(priors))
    for subkey, prior_node in zip(keys, priors):
        plot_prior(subkey, prior_node, output_directory, num_samples)


def get_property_values(shape_parameters, name):
    values = []
    for type_name, type_values in shape_parameters.items():
        for property_name, property_value in type_values.items():
            if property_name == name:
                values.append(property_value)
    return np.array(values)


def fit_gaussian_mixture(seed, samples, unidimensional=True):
    _, fit_key = jax.random.split(jax.random.PRNGKey(seed))
    flat_samples = samples.reshape(-1, 1) if unidimensional else samples
    model = GMM(2, covariance="diag", name="gmm")
    return model.fit(
        fit_key,
        jp.asarray(flat_samples),
        method="em",
        num_iters=200,
        regularization=1e-3,
    )


def build_affine_bijector(x):
    return tfb.Chain([tfb.Shift(x.shift), tfb.Scale(x.scale)])


def build_shift_prior(mean, scale):
    mean, scale = jp.array(mean), jp.array(scale)
    lower, upper = jp.array([-0.14, -0.14]), jp.array([0.14, 0.14])
    distribution = tfd.TruncatedNormal(mean, scale, lower, upper)
    bijector = tfb.Chain([tfb.Shift(mean), tfb.Scale(scale)])
    return paz.Prior(distribution, name="shift", bijector=bijector)


def build_theta_prior(mean, concentration):
    distribution = tfd.VonMises(mean, concentration)
    bijector = tfb.Chain([tfb.Sigmoid(-jp.pi, jp.pi), tfb.Scale(jp.pi / 2.0)])
    return paz.Prior(distribution, name="theta", bijector=bijector)


def build_scale_prior(mean, variance):
    mean, variance = jp.array(mean), jp.array(variance)
    variance = jp.log((variance / mean**2) + 1)
    mean = jp.log(mean) - (variance / 2)
    scale = jp.sqrt(variance)
    distribution = tfd.LogNormal(mean, scale)
    bijector = tfb.Chain([tfb.Softplus(), tfb.Shift(mean), tfb.Scale(scale)])
    return paz.Prior(distribution, name="scale", bijector=bijector)


def build_classes_prior(probabilities, temperature):
    distribution = tfd.RelaxedOneHotCategorical(
        temperature, probs=probabilities
    )
    bijector = tfb.Chain([tfb.SoftmaxCentered(), tfb.Scale(3.0)])
    return paz.Prior(distribution, name="classes", bijector=bijector)


def save_priors(priors, output_directory, filename):
    pgm = paz.PGM(priors, priors, "optimized_priors")
    path = Path(output_directory) / filename
    paz.inference.save(pgm, path, overwrite=True)
    return path


def load_priors(path):
    return paz.inference.load(path).inputs


def find_latest_directory(wildcard):
    matches = [
        Path(path) for path in glob.glob(str(wildcard)) if Path(path).is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No previous optimization found for wildcard: {wildcard}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def require_files(directory_path, filenames):
    missing = []
    for filename in filenames:
        path = Path(directory_path) / filename
        candidates = [path]
        if path.suffix != ".pkl":
            candidates.append(path.with_suffix(".pkl"))
        if not any(candidate.exists() for candidate in candidates):
            missing.append(filename)
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files in {directory_path}: {missing_list}"
        )


root = build_directory(Path(args.root) / args.dataset_name, args.label)
paz.file.write_json(args.__dict__, root / "parameters.json")
wildcard = Path(args.root) / args.dataset_name / args.parameters_wildcard
key = jax.random.PRNGKey(args.seed)
material_priors = []

optimization_directory = find_latest_directory(wildcard)
require_files(
    optimization_directory,
    [
        args.materials_filename,
        args.parameters_filename,
        "floor.pkl",
        "lights.pkl",
    ],
)
shape_parameters = load_parameters(wildcard, args.materials_filename)
print(f"Found previous optimization at: {optimization_directory}")
print("Optimizing bijectors for material properties...")

for name in ["ambient", "diffuse", "specular", "shininess", "color"]:
    values = get_property_values(shape_parameters, name)
    target_dist = fit_gaussian_mixture(args.seed, values, name != "color")
    key, key_g = jax.random.split(key)
    shift_0 = 0.0 if name != "color" else jp.full(3, 0.0)
    scale_0 = 1.0 if name != "color" else jp.full(3, 1.0)
    x_0 = BijectorParams(shift_0, scale_0)
    distribution = tfd.Normal(shift_0, scale_0)
    bijector = build_affine_bijector(x_0)
    prior = paz.Prior(distribution, name=name, bijector=bijector)
    fitted_prior, losses = prior.fit_bijector(
        key_g,
        target_dist,
        num_samples=args.num_samples,
        num_steps=args.gradient_steps,
        return_losses=True,
    )
    write_losses(losses, root, f"loss_{name}.pdf")
    material_priors.append(fitted_prior)


dataset_metadata = paz.datasets.fsclvr.parse_metadata(args.dataset_name)
camera_origin = jp.array(dataset_metadata["camera_origin"])
H, W = dataset_metadata["image_shape"]
y_FOV = dataset_metadata["y_FOV"]

optimization_metadata = load_parameters(wildcard, args.parameters_filename)
viewport_factor = optimization_metadata["viewport_factor"]
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]

floor_filename = optimization_directory / "floor.pkl"
lights_filename = optimization_directory / "lights.pkl"
with floor_filename.open("rb") as filedata:
    floor = pickle.load(filedata)
with lights_filename.open("rb") as filedata:
    lights = pickle.load(filedata)
print(f"\nLoaded floor and lights from: {optimization_directory}")

camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = paz.SE3.view_transform(
    camera_origin, camera_target, camera_upward
)
_render = build_render_function(
    image_shape, y_FOV, camera_pose, lights, args.shadows
)
render = build_observation_model(_render, floor)

print("Building render function and observation model...")

geometry_priors = [
    build_shift_prior(args.shift_mean, args.shift_scale),
    build_theta_prior(args.theta_mean, args.theta_concentration),
    build_scale_prior(args.scale_mean, args.scale_scale),
    build_classes_prior(args.class_probabilities, args.classes_temperature),
]

all_priors = material_priors + geometry_priors
priors_path = save_priors(all_priors, root, args.prior_filename)
print(f"Saved optimized priors to: {priors_path}")
all_priors = load_priors(priors_path)

print(f"\nAll prior names: {[p.name for p in all_priors]}")
print(f"Number of priors: {len(all_priors)}")

print(f"\nGenerating {args.num_samples} samples for prior visualization...")
plot_priors(key, all_priors, root, args.num_samples)
print(f"Saved prior plots to: {root}")

image_directory = root / "prior_predictive_samples"
directory.make(image_directory)

print(f"\nGenerating {args.num_images} prior predictive samples...")
for image_arg, subkey in enumerate(jax.random.split(key, args.num_images)):
    subkeys = jax.random.split(subkey, len(all_priors))
    sample_dict = {}
    for prior, subkey_i in zip(all_priors, subkeys):
        sample = prior.sample(subkey_i, 1)
        sample_dict[prior.name] = jp.squeeze(sample)
    image, _ = render(sample_dict)
    filename = f"{image_arg:02d}_prior_predictive_sample.png"
    write_image(image, image_directory, filename)
    if (image_arg + 1) % 10 == 0:
        print(f"  Generated {image_arg + 1}/{args.num_images} images")

print(f"\nSaved prior predictive samples to: {image_directory}")
print("Done!")
