import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import pandas

import paz
from observation_model import build_display_model, build_observation_model
from observation_model import build_render_function, hide_shape, parse_summary


def load_scene(parameters):
    directory = Path(parameters["scene_directory"])
    scene = json.load(open(directory / parameters["params_filename"], "r"))
    lights = load_pickle(directory / "lights.pkl")
    floor = load_pickle(directory / "floor.pkl")
    return scene, lights, floor


def load_pickle(filepath):
    with open(filepath, "rb") as filedata:
        return pickle.load(filedata)


def build_camera_pose(camera_origin):
    upward = jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(camera_origin, jp.zeros(3), upward)


def build_fitted_model(image_shape, y_FOV, pose, lights, floor, shadows):
    """Renders as the likelihood saw the scene while sampling."""
    intrinsics = paz.graphics.camera.compute_intrinsics(y_FOV, *image_shape)
    render_args = (image_shape, intrinsics, pose, lights, shadows)
    return build_observation_model(build_render_function(*render_args), floor)


def write_estimates(directory, summary, true_image, fitted_model, display):
    for statistic in ["mean", "median"]:
        sample = parse_summary(summary, statistic)
        write_image(display(sample), directory, f"{statistic}_point.png")
        fitted_image, _ = fitted_model(sample)
        pair = jp.concatenate([true_image, fitted_image], axis=1)
        write_image(pair, directory, f"true_pred_{statistic}_image.png")
        error = (true_image - fitted_image) ** 2
        write_image(error, directory, f"error_{statistic}_image.png")


def write_ground_images(directory, summary, ground_display, camera_origin):
    sample = parse_summary(summary, "median")
    write_image(ground_display(sample), directory, "ground_image.png")
    floor_image = ground_display(hide_shape(sample, camera_origin))
    write_image(floor_image, directory, "ground_floor.png")


def write_samples(key, directory, trace, display, num_samples):
    samples = draw_samples(key, trace, num_samples)
    for sample_arg in range(num_samples):
        sample = {name: values[sample_arg] for name, values in samples.items()}
        filename = f"posterior_sample_{sample_arg:02d}.png"
        write_image(display(sample), directory, filename)


def draw_samples(key, trace, num_samples):
    num_draws = trace["theta"].size
    indices = jax.random.choice(key, num_draws, (num_samples,))
    samples = {}
    for name, values in trace.items():
        draws = jp.array(values).reshape(num_draws, *values.shape[2:])
        samples[name] = draws[indices]
    return samples


def write_image(image, directory, filename):
    image = paz.image.denormalize(jp.clip(image, 0.0, 1.0))
    paz.image.write(Path(directory) / filename, image)


if __name__ == "__main__":
    description = "Render the estimates and samples of a learn_concept run"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()

    directory = Path(args.directory)
    parameters = json.load(open(directory / "parameters.json", "r"))
    scene, lights, floor = load_scene(parameters)
    metadata = paz.datasets.fsclvr.parse_metadata(scene["dataset_name"])
    H, W = metadata["image_shape"]
    factor = scene["viewport_factor"]
    fitted_shape = [int(H * factor), int(W * factor)]
    camera_origin = jp.array(metadata["camera_origin"])
    pose = build_camera_pose(camera_origin)
    scene_args = (metadata["y_FOV"], pose, lights, floor)
    fitted_args = (fitted_shape, *scene_args, parameters["shadows"])
    fitted_model = build_fitted_model(*fitted_args)
    display = build_display_model((H, W), *scene_args)
    ground_display = build_display_model((2 * H, 2 * W), *scene_args)

    summary = pandas.read_csv(directory / "summary.csv", index_col=0)
    true_image = paz.image.load(directory / "true_image.png") / 255.0
    estimate_args = (directory, summary, true_image, fitted_model, display)
    write_estimates(*estimate_args)
    write_ground_images(directory, summary, ground_display, camera_origin)
    sample_directory = paz.directory.make(directory / "posterior_samples")
    trace = load_pickle(directory / "trace.pkl")
    key = jax.random.PRNGKey(args.seed)
    write_samples(key, sample_directory, trace, display, args.num_samples)
