import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np

import paz
import figures
from draw_program import write_posterior_program
from observation_model import parse_true_values, sample_priors
from observation_model import project_ground_points


def load_image(filepath):
    return plt.imread(filepath)[..., :3]


def load_images(directory):
    paths = sorted((directory / "posterior_samples").glob("*.png"))[:9]
    samples = np.array([load_image(path) for path in paths])
    observation = load_image(directory / "true_image.png")
    estimate = load_image(directory / "median_point.png")
    return observation, estimate, figures.build_mosaic(samples, (3, 3))


def build_ground_projector(parameters, image_shape):
    scene_directory = Path(parameters["scene_directory"])
    scene_path = scene_directory / parameters["params_filename"]
    scene = json.load(open(scene_path, "r"))
    metadata = paz.datasets.fsclvr.parse_metadata(scene["dataset_name"])
    camera_origin = jp.array(metadata["camera_origin"])
    upward = jp.array([0.0, 1.0, 0.0])
    pose = paz.SE3.view_transform(camera_origin, jp.zeros(3), upward)
    camera_args = (image_shape, metadata["y_FOV"], pose)
    return paz.lock(project_ground_points, *camera_args)


def load_trace(directory):
    with open(directory / "trace.pkl", "rb") as filedata:
        trace = pickle.load(filedata)
    return {name: np.asarray(values) for name, values in trace.items()}


if __name__ == "__main__":
    description = "Redraw the posterior figures of a learn_concept run"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()

    directory = Path(args.directory)
    parameters = json.load(open(directory / "parameters.json", "r"))
    priors_directory = Path(parameters["priors_directory"])
    priors_path = priors_directory / parameters["prior_filename"]
    priors = paz.inference.load(priors_path).inputs
    key = jax.random.PRNGKey(args.seed)
    prior_samples, _ = sample_priors(key, priors, 20_000)
    label = json.load(open(directory / "label.json", "r"))
    images = load_images(directory)
    figures_directory = paz.directory.make(directory / "figures")
    figures.configure()
    figure_args = (prior_samples, parse_true_values(label), images)
    trace = load_trace(directory)
    figures.write_posterior_figures(figures_directory, trace, *figure_args)
    ground_image = load_image(directory / "ground_image.png")
    floor_image = load_image(directory / "ground_floor.png")
    project = build_ground_projector(parameters, ground_image.shape[:2])
    shift_samples = figures.flatten(trace["shift"])
    ground_args = (shift_samples, ground_image, floor_image, project)
    ground_figure = figures.plot_ground_shift(*ground_args)
    figures.save(ground_figure, figures_directory, "shift_on_ground")
    sample_paths = sorted((directory / "posterior_samples").glob("*.png"))
    observation_path = directory / "true_image.png"
    program_args = (figures_directory, observation_path, sample_paths)
    write_posterior_program(*program_args)
