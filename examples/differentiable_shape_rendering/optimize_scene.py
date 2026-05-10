import argparse
import os
from collections import namedtuple
from functools import partial
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
import jax.numpy as jp
import optax

import paz
import paz.utils.plot as plot
from paz.optimization.history import trim_trace

MIN_STEP_DIGITS = 4
LOSS_CSV_HEADER = ["step", "mse"]
MATERIAL_KEYS = ("ambient", "diffuse", "specular", "shininess")
MATERIAL_KEYS += ("reflective", "transparency", "refractive_index")

STEP_DIRECTORY_FIELDS = ("prediction_dir", "scene_dir", "step_digits")
StepDirectory = namedtuple("StepDirectory", STEP_DIRECTORY_FIELDS)
OPTIMIZATION_FIELDS = ("parameters", "history", "initial_prediction")
OptimizationOutputs = namedtuple("OptimizationOutputs", OPTIMIZATION_FIELDS)

world_up = jp.array([0.0, 1.0, 0.0])
light_position = jp.array([2.4, 3.8, -2.0])
lights = paz.graphics.PointLight(jp.full(3, 1.32), light_position)

camera_shift = 1.8 * jp.array([-1.0, 1.0, -2.0])
world_to_camera = paz.SE3.view_transform(camera_shift, jp.zeros(3), world_up)

H, W, y_fov, resize_factor = image_size = 1024, 1024, jp.pi / 5.0, 1
shading = dict(mask=None, shadows=True, lights=lights)
chunking = dict(tiles=(1, 1), chunk_size=2**10)
render_kwargs = paz.merge_dicts(shading, chunking)
rays = paz.graphics.camera.build_rays(image_size, y_fov, world_to_camera)
render_args = ((H, W), y_fov, world_to_camera)
render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))

floor_half_height = 0.1
floor_size = paz.SE3.scaling(jp.array([1.5, floor_half_height, 1.5]))

sphere_radius = 0.35
sphere_scale = paz.SE3.scaling(sphere_radius)
sphere_y = floor_half_height + sphere_radius + 0.002
sphere_shift = paz.SE3.translation(jp.array([1.00, sphere_y, 0.7]))
sphere_transform = sphere_shift @ sphere_scale

cone_size = 0.45
cone_scale = paz.SE3.scaling(cone_size)
cone_shift = paz.SE3.translation(jp.array([-0.6, cone_size, -0.4]))

default_color = 0.8 * jp.ones(3)
default_material = paz.graphics.Material(default_color, 0.85, 0.1, 0.0, 100.0)
material_init = {n: default_material for n in ("floor", "sphere", "cone")}


def build_scene(parameters):
    floor = paz.graphics.Cube(floor_size, parameters["floor"])
    floor = floor._replace(transform=floor_size)
    sphere = paz.graphics.Sphere(sphere_transform, parameters["sphere"])
    cone = paz.graphics.Cone(cone_shift @ cone_scale, parameters["cone"])
    return paz.graphics.Scene([floor, sphere, cone])


def render_prediction(parameters):
    image, _ = render(scene=build_scene(parameters))
    return image


def build_loss_fn(target_image):
    def loss_fn(parameters):
        prediction = render_prediction(parameters)
        return jp.mean((prediction - target_image) ** 2)

    return loss_fn


def format_step_number(step_number, step_digits):
    return f"{step_number:0{step_digits}d}"


def build_step_prediction_path(step_directory, step_number):
    step_text = format_step_number(step_number, step_directory.step_digits)
    return step_directory.prediction_dir / f"prediction_step_{step_text}.png"


def build_step_scene_path(step_directory, step_number):
    step_text = format_step_number(step_number, step_directory.step_digits)
    return step_directory.scene_dir / f"scene_step_{step_text}"


def should_log_step(step_number, log_every, num_steps):
    return (step_number % log_every == 0) or (step_number == num_steps)


def should_save_step(step_number, save_every):
    return save_every > 0 and (step_number % save_every == 0)


def build_step_callback(run_arguments, step_directory):
    def callback(step_num, parameters, loss, _metrics):
        log_args = (step_num, run_arguments.log_every, run_arguments.num_steps)
        if should_log_step(*log_args):
            step_text = format_step_number(step_num, step_directory.step_digits)
            print(f"step={step_text} mse={float(loss):.8f}")
        if should_save_step(step_num, run_arguments.save_every):
            save_step_prediction(parameters, step_num, step_directory)
        if should_save_step(step_num, run_arguments.save_scene_every):
            save_step_scene(parameters, step_num, step_directory)

    return callback


def save_normalized_image(filepath, image):
    image = jp.clip(image, 0.0, 1.0)
    image = paz.image.denormalize(image)
    paz.image.write(str(filepath), image)


def save_initial_prediction(parameters, output_dir, step_directory):
    prediction = render_prediction(parameters)
    filepath = output_dir / "initial_prediction.png"
    save_normalized_image(filepath, prediction)
    step_filepath = build_step_prediction_path(step_directory, 0)
    save_normalized_image(step_filepath, prediction)
    return prediction


def save_step_prediction(parameters, step_number, step_directory):
    prediction = render_prediction(parameters)
    filepath = build_step_prediction_path(step_directory, step_number)
    save_normalized_image(filepath, prediction)


def save_step_scene(parameters, step_number, step_directory):
    scene = build_scene(parameters)
    output_path = build_step_scene_path(step_directory, step_number)
    paz.graphics.save(str(output_path), scene)


def build_step_directory(output_dir, num_steps):
    prediction_dir = Path(paz.directory.make(output_dir / "prediction_steps"))
    scene_dir = Path(paz.directory.make(output_dir / "scene_steps"))
    step_digits = max(MIN_STEP_DIGITS, len(str(num_steps)))
    return StepDirectory(prediction_dir, scene_dir, step_digits)


def optimize_materials(parameters, target_image, output_dir, run_arguments):
    optimizer = optax.adam(learning_rate=run_arguments.learning_rate)
    directory = build_step_directory(output_dir, run_arguments.num_steps)
    initial = save_initial_prediction(parameters, output_dir, directory)
    save_step_scene(parameters, 0, directory)
    loss_fn = build_loss_fn(target_image)
    callbacks = [build_step_callback(run_arguments, directory)]
    args = (parameters, loss_fn, optimizer, run_arguments.num_steps)
    _, parameters, history = paz.minimize(*args, callbacks=callbacks)
    return OptimizationOutputs(parameters, history, initial)


def assert_target_image_exists(filepath):
    if filepath.exists():
        return
    message = f"Target image '{filepath}' not found. Run build_true_image.py."
    raise FileNotFoundError(message)


def build_target_image(filepath, expected_shape):
    filepath = Path(filepath)
    assert_target_image_exists(filepath)
    image = paz.image.load(str(filepath))
    if image.shape[:2] != expected_shape:
        image = paz.image.resize_opencv(image, expected_shape)
    image = paz.image.normalize(image)
    return jp.array(image, dtype=jp.float32)


RUN_CONFIG_KEYS = ("target_image_path", "num_steps", "learning_rate")
RUN_CONFIG_KEYS += ("log_every", "save_every", "save_scene_every")


def write_run_config(output_dir, run_arguments):
    config = {key: getattr(run_arguments, key) for key in RUN_CONFIG_KEYS}
    config["image_shape"] = [H, W]
    config["y_fov"] = float(y_fov)
    config["sphere_y"] = float(sphere_y)
    config["uses_default_zero_patterns"] = True
    paz.file.write_json(config, output_dir / "config.json")


def serialize_material(material):
    serialized = {key: float(getattr(material, key)) for key in MATERIAL_KEYS}
    serialized["color"] = [float(value) for value in material.color]
    return serialized


def save_materials(filepath, materials):
    items = materials.items()
    serializable = {name: serialize_material(m) for name, m in items}
    paz.file.write_json(serializable, str(filepath))


def write_loss_csv(filepath, history):
    losses = jax.device_get(trim_trace(history).losses)
    with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
        writer = paz.file.csv.writer(csv_file)
        writer.writerow(LOSS_CSV_HEADER)
        for i, loss in enumerate(losses):
            writer.writerow([i + 1, float(loss)])


def save_loss_plot(filepath, history):
    losses = jax.device_get(trim_trace(history).losses)
    if len(losses) == 0:
        return
    steps = list(range(1, len(losses) + 1))
    plot.configure(fontsize=12, linewidth=2.0)
    figure, axis = plot.subplots(1, 1, figsize=(8, 4))
    plot.line(steps, losses, axis=axis, color=plot.DEFAULT_PALETTE.primary)
    plot.set_labels(axis, x="Optimization Step", y="MSE")
    plot.clean(axis)
    plot.save(figure, str(filepath))


def build_loss_summary(history):
    losses = jax.device_get(trim_trace(history).losses)
    summary = {}
    summary["initial_mse"] = float(losses[0])
    summary["final_mse"] = float(losses[-1])
    summary["best_mse"] = float(losses.min())
    summary["num_steps"] = int(len(losses))
    return summary


def save_comparison_image(output_dir, target, outputs, optimized):
    images = [target, outputs.initial_prediction, optimized]
    comparison = jp.concatenate(images, axis=1)
    filepath = output_dir / "target_initial_optimized.png"
    save_normalized_image(filepath, comparison)


def save_final_outputs(output_dir, target, outputs):
    parameters = outputs.parameters
    history = outputs.history
    optimized = render_prediction(parameters)
    scene = build_scene(parameters)
    paz.graphics.save(str(output_dir / "optimized_scene"), scene)
    save_normalized_image(output_dir / "optimized_prediction.png", optimized)
    save_materials(output_dir / "optimized_materials.json", parameters)
    write_loss_csv(output_dir / "loss_history.csv", history)
    save_loss_plot(output_dir / "loss_curve.png", history)
    summary = build_loss_summary(history)
    paz.file.write_json(summary, output_dir / "loss_summary.json")
    save_comparison_image(output_dir, target, outputs, optimized)


description = "Optimize only materials to match true_image.png"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--target-image-path", type=str, default="true_image.png")
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--save-every", type=int, default=10)
parser.add_argument("--save-scene-every", type=int, default=10)
parser.add_argument("--experiments-root", type=str, default="experiments")
parser.add_argument("--experiment-label", default="optimize_materials_only")
run_arguments = parser.parse_args()

output_args = (run_arguments.experiments_root, run_arguments.experiment_label)
output_dir = Path(paz.directory.make_timestamped(*output_args))
print(f"output_dir={output_dir}")
print(f"learning_rate={run_arguments.learning_rate:.6f}")
target_image = build_target_image(run_arguments.target_image_path, (H, W))
parameters = material_init
write_run_config(output_dir, run_arguments)
args = (parameters, target_image, output_dir, run_arguments)
optimization_outputs = optimize_materials(*args)
save_final_outputs(output_dir, target_image, optimization_outputs)
