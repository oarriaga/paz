import os
import pickle
import argparse
from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np
from sklearn.mixture import GaussianMixture as SKLGaussianMixture
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
import arviz as az

tfd = tfp.distributions
tfb = tfp.bijectors

import paz
from paz.backend import directory, file
from observation_model import build_observation_model, build_render_function

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
parser.add_argument("--prior_filename", default="prior.json")
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


DANDELION = [0.992, 0.737, 0.258]
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
    import json

    filepath = directory.find_latest(wildcard)
    filepath = os.path.join(filepath, filename)

    # Try .json first, then .pkl
    if os.path.exists(filepath):
        with open(filepath, "r") as filedata:
            return json.load(filedata)
    elif os.path.exists(filepath.replace(".json", ".pkl")):
        import pickle
        with open(filepath.replace(".json", ".pkl"), "rb") as filedata:
            return pickle.load(filedata)
    else:
        raise FileNotFoundError(f"Could not find {filename} or {filename.replace('.json', '.pkl')}")


def build_directory(root, label):
    return directory.make_timestamped(root, label)


def write_image(image, output_directory, filename):
    if not filename.endswith(".png"):
        raise ValueError(f"Filename {filename} missing extension .png")
    filepath = os.path.join(output_directory, filename)
    plt.imsave(filepath, image)


def write_losses(losses, output_directory, filename):
    figure, axis = plt.subplots()
    axis.plot(losses, color=DANDELION)
    axis.set_ylabel("loss")
    axis.set_xlabel("step")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fullpath = os.path.join(output_directory, filename)
    if not fullpath.endswith(".pdf"):
        raise ValueError(f"Filename {fullpath} missing extension .pdf")
    figure.savefig(fullpath, bbox_inches="tight")
    plt.close()


def plot_forward_samples(forward_samples, name, output_directory):
    figure, axis = plt.subplots()
    az.plot_dist(
        forward_samples,
        color=DANDELION,
        ax=axis,
        plot_kwargs={"linewidth": 5},
        fill_kwargs={"alpha": 0.4},
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20
    axis.set_ylabel("density")
    axis.set_xlabel(NAME_TO_TEX.get(name, name))
    filename = os.path.join(output_directory, f"prior_forward_{name}.pdf")
    figure.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_inverse_samples(inverse_samples, normals_samples, name, output_directory):
    figure, axis = plt.subplots()
    kwargs = {"plot_kwargs": {"linewidth": 5}, "fill_kwargs": {"alpha": 0.4}}
    az.plot_dist(normals_samples, color="r", ax=axis, label="normal", **kwargs)
    az.plot_dist(inverse_samples, color="b", ax=axis, label="inverse", **kwargs)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20
    axis.legend(prop={"size": 10}, frameon=False)
    axis.set_ylabel("density")
    axis.set_xlabel(NAME_TO_TEX.get(name, name))
    filename = os.path.join(output_directory, f"prior_inverse_{name}.pdf")
    figure.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_prior(key, prior_node, output_directory, num_samples=10_000):
    key_0, key_1, key_2 = jax.random.split(key, 3)
    forward_samples = prior_node.sample(key_0, num_samples)
    inverse_samples = prior_node.sample_inverse(key_1, num_samples)
    normals_samples = tfd.Normal(0.0, 1.0).sample(num_samples, seed=key_2)
    name = prior_node.name

    if name == "shift":
        for arg, shift_name in enumerate(["x-translation", "y-translation"]):
            plot_forward_samples(forward_samples[:, arg], shift_name, output_directory)
            plot_inverse_samples(
                inverse_samples[:, arg], normals_samples, shift_name, output_directory
            )
    elif name == "color":
        for arg, channel_name in enumerate(["R-color", "G-color", "B-color"]):
            plot_forward_samples(
                forward_samples[:, arg], channel_name, output_directory
            )
            plot_inverse_samples(
                inverse_samples[:, arg], normals_samples, channel_name, output_directory
            )
    elif name == "scale":
        for arg, scale_name in enumerate(["x-scale", "y-scale", "z-scale"]):
            plot_forward_samples(forward_samples[:, arg], scale_name, output_directory)
            plot_inverse_samples(
                inverse_samples[:, arg], normals_samples, scale_name, output_directory
            )
    else:
        plot_forward_samples(forward_samples, name, output_directory)
        plot_inverse_samples(inverse_samples, normals_samples, name, output_directory)


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


def extract_affine_params(bijector):
    shift = bijector.bijectors[0].shift
    scale = bijector.bijectors[1].scale
    return {"shift": jp.asarray(shift).tolist(), "scale": jp.asarray(scale).tolist()}


def get_mixture_parameters(model, unidimensional=True):
    weights = model.weights_.tolist()
    if unidimensional:
        mean = model.means_[:, 0].tolist()
        stdv = np.sqrt(model.covariances_[:, 0]).tolist()
    else:
        mean = np.moveaxis(np.array(model.means_), 1, 0).tolist()
        stdv = np.moveaxis(np.sqrt(model.covariances_), 1, 0).tolist()
    return {"weights": weights, "mean": mean, "stdv": stdv}


def fit_gaussian_mixture(seed, samples, unidimensional=True):
    kwargs = {
        "covariance_type": "diag",
        "max_iter": 1000,
        "reg_covar": 1e-3,
        "n_init": 500,
        "tol": 1e-4,
        "random_state": seed,
    }
    model = SKLGaussianMixture(2, **kwargs)
    if unidimensional:
        samples = samples.reshape(-1, 1)
    model = model.fit(samples)
    return get_mixture_parameters(model, unidimensional)


def build_affine_bijector(x):
    return tfb.Chain([tfb.Shift(x.shift), tfb.Scale(x.scale)])


def build_gmm(weights, mean, stdv):
    return tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights), tfd.Normal(loc=mean, scale=stdv)
    )


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
    distribution = tfd.RelaxedOneHotCategorical(temperature, probs=probabilities)
    bijector = tfb.Chain([tfb.SoftmaxCentered(), tfb.Scale(3.0)])
    return paz.Prior(distribution, name="classes", bijector=bijector)


def build_material_prior(name, mixture_params, bijector_params):
    mixture_dist = build_gmm(**mixture_params)
    bijector = tfb.Invert(build_affine_bijector(BijectorParams(**bijector_params)))
    return paz.Prior(mixture_dist, name=name, bijector=bijector)


def load_material_priors(wildcard, filename):
    import json

    prior_directory = directory.find_latest(wildcard)
    filedata = open(os.path.join(prior_directory, filename), "r")
    priors_params = json.load(filedata)
    priors = []
    for name, prior_params in priors_params.items():
        mixture_params = prior_params["mixture"]
        bijector_params = prior_params["bijector"]
        prior = build_material_prior(name, mixture_params, bijector_params)
        priors.append(prior)
    return priors


root = build_directory(os.path.join(args.root, args.dataset_name), args.label)
file.write_json(args.__dict__, os.path.join(root, "parameters.json"))
priors = {}
wildcard = os.path.join(args.root, args.dataset_name, args.parameters_wildcard)

try:
    shape_parameters = load_parameters(wildcard, args.materials_filename)
    print(f"Found previous optimization at: {directory.find_latest(wildcard)}")
    print("Optimizing bijectors for material properties...")

    key = jax.random.PRNGKey(args.seed)
    for name in ["ambient", "diffuse", "specular", "shininess", "color"]:
        one_dim = True if name != "color" else False
        values = get_property_values(shape_parameters, name)
        mixture_params = fit_gaussian_mixture(args.seed, values, one_dim)
        key, key_n, key_g = jax.random.split(key, 3)
        shift_0 = 0.0 if name != "color" else jp.full(3, 0.0)
        scale_0 = 1.0 if name != "color" else jp.full(3, 1.0)
        x_0 = BijectorParams(shift_0, scale_0)
        distribution = tfd.Normal(shift_0, scale_0)
        bijector = build_affine_bijector(x_0)
        target_distribution = build_gmm(**mixture_params)
        prior = paz.Prior(distribution, name=name, bijector=bijector)
        fitted_prior, losses = prior.fit_bijector(
            key_g,
            target_distribution,
            num_samples=args.num_samples,
            num_steps=args.gradient_steps,
            return_losses=True,
        )
        write_losses(losses, root, f"loss_{name}.pdf")
        bijector_params = extract_affine_params(fitted_prior.metadata.bijector)
        priors[name] = {"mixture": mixture_params, "bijector": bijector_params}
    file.write_json(priors, os.path.join(root, args.prior_filename))
    print(f"Saved optimized bijector priors to: {os.path.join(root, args.prior_filename)}")

except (ValueError, FileNotFoundError) as e:
    print(f"\nWarning: Could not find previous SCENE-OPTIMIZATION run.")
    print(f"Searched for: {wildcard}")
    print(f"Error: {e}")
    print("\nSkipping bijector optimization and prior predictive sampling.")
    print("\nTo run the full script:")
    print(f"  1. First run: python optimize_scene.py --dataset_name {args.dataset_name}")
    print(f"  2. Then run: python optimize_bijectors.py --dataset_name {args.dataset_name}")
    print(f"\nParameters saved to: {os.path.join(root, 'parameters.json')}")
    import sys
    sys.exit(0)


dataset_metadata = paz.datasets.fsclvr.parse_metadata(args.dataset_name)
camera_origin = jp.array(dataset_metadata["camera_origin"])
H, W = dataset_metadata["image_shape"]
y_FOV = dataset_metadata["y_FOV"]

wildcard = os.path.join(args.root, args.dataset_name, args.parameters_wildcard)
optimization_metadata = load_parameters(wildcard, args.parameters_filename)
viewport_factor = optimization_metadata["viewport_factor"]
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]

optimization_directory = directory.find_latest(wildcard)
floor_filename = os.path.join(optimization_directory, "floor.pkl")
floor = pickle.load(open(floor_filename, "rb"))

lights_filename = os.path.join(optimization_directory, "lights.pkl")
lights = pickle.load(open(lights_filename, "rb"))

print(f"\nLoaded floor and lights from: {optimization_directory}")

camera_origin = jp.array(camera_origin)
camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = paz.SE3.view_transform(camera_origin, camera_target, camera_upward)
_render = build_render_function(image_shape, y_FOV, camera_pose, lights, args.shadows)
render = build_observation_model(_render, floor)

print("Building render function and observation model...")

priors_wildcard = os.path.join(args.root, args.dataset_name, "*" + args.label)
material_priors = load_material_priors(priors_wildcard, args.prior_filename)
geometry_priors = [
    build_shift_prior(args.shift_mean, args.shift_scale),
    build_theta_prior(args.theta_mean, args.theta_concentration),
    build_scale_prior(args.scale_mean, args.scale_scale),
    build_classes_prior(args.class_probabilities, args.classes_temperature),
]

all_priors = material_priors + geometry_priors

print(f"\nAll prior names: {[p.name for p in all_priors]}")
print(f"Number of priors: {len(all_priors)}")

print(f"\nGenerating {args.num_samples} samples for prior visualization...")
plot_priors(key, all_priors, root, args.num_samples)
print(f"Saved prior plots to: {root}")

image_directory = os.path.join(root, "prior_predictive_samples")
directory.make(image_directory)

print(f"\nGenerating {args.num_images} prior predictive samples...")
for image_arg, subkey in enumerate(jax.random.split(key, args.num_images)):
    subkeys = jax.random.split(subkey, len(all_priors))
    sample_dict = {}
    for prior, subkey_i in zip(all_priors, subkeys):
        sample = prior.sample(subkey_i, 1)
        sample_dict[prior.name] = jp.squeeze(sample)
    image, depth = render(sample_dict)
    filename = f"{image_arg:02d}_prior_predictive_sample.png"
    write_image(image, image_directory, filename)
    if (image_arg + 1) % 10 == 0:
        print(f"  Generated {image_arg + 1}/{args.num_images} images")

print(f"\nSaved prior predictive samples to: {image_directory}")
print("Done!")
