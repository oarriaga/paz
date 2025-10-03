from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
import argparse
import paz
import jax
import jax.numpy as jp
import optax

from paz.datasets.fsclvr import parse_label
from paz.graphics import (
    PointLight,
    Material,
    Pattern,
    Shape,
    NO_PATTERN,
    PLANAR_PATTERN,
    PLANE,
    SPHERE,
    CUBE,
    CYLINDER,
)


parser = argparse.ArgumentParser(description="scene optimization")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--dataset_name", default="plain", type=str)
parser.add_argument("--dataset_path", default="datasets", type=str)
parser.add_argument("--shapes_directory", default="shapes", type=str)
parser.add_argument("--images_directory", default="images", type=str)
# parser.add_argument("--viewport_factor", default=0.25, type=float)
parser.add_argument("--viewport_factor", default=1.0, type=float)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--split", default="train", type=str)
parser.add_argument("--label", default="SCENE-OPTIMIZATION", type=str)
parser.add_argument("--num_lights", default=4, type=int)
parser.add_argument("--pattern_shape", nargs="+", default=[200, 200, 3])
parser.add_argument("--outer_epochs", default=7, type=int)
parser.add_argument("--inner_epochs", default=5, type=int)
parser.add_argument("--shadows", default=True, type=bool)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--ambient", default=0.1, type=float)
parser.add_argument("--diffuse", default=0.9, type=float)
parser.add_argument("--specular", default=0.5, type=float)
parser.add_argument("--shininess", default=4.0, type=float)
parser.add_argument("--min_intensity", default=4.0, type=float)
parser.add_argument("--max_intensity", default=4.0, type=float)
args = parser.parse_args()


FLOOR = namedtuple("FLOOR", ["color", "ambient", "diffuse", "image"])
LANDSCAPE_VARIABLE = namedtuple("SCENE", ["floor", "lights"])
MATERIAL_FIELDS = ["color", "ambient", "diffuse", "specular", "shininess"]
MATERIAL_VARIABLE = namedtuple("MATERIAL_VARIABLE", MATERIAL_FIELDS)
VARIABLE_STATE = namedtuple("VARIABLE_STATE", ["optimizer_state", "variable"])
SHAPE_TYPES = [SPHERE, CUBE, CYLINDER]
NAME_TO_TYPE = dict(zip(["sphere", "cube", "cylinder"], SHAPE_TYPES))
ARG_TO_TYPE = dict(zip([0, 1, 2], SHAPE_TYPES))


def build_floor(floor):
    pattern = Pattern(jp.eye(4), PLANAR_PATTERN, floor.image)
    material = Material(floor.color, floor.ambient, floor.diffuse, 0.0, 200.0)
    return Shape(jp.eye(4), PLANE, material, pattern)


def build_render(origin, H, W, y_FOV, shadows):
    origin = jp.array(camera_origin)
    target = jp.array([0.0, 0.0, 0.0])
    upward = jp.array([0.0, 1.0, 0.0])
    camera_pose = paz.SE3.view_transform(origin, target, upward)
    rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
    args = (image_shape, camera_pose, rays)
    return paz.partial(paz.graphics.render, *args, shadows=shadows, mask=None)


def Scene(image_shape, y_FOV, camera_origin, shadows):
    render = build_render(camera_origin, *image_shape, y_FOV, shadows)

    def render_scene(shape, scene):
        floor = build_floor(scene.floor)
        shape = paz.graphics.Scene([shape, floor])
        image, depth = render(scene=shape, lights=scene.lights)
        return jp.clip(image, 0.0, 1.0), depth

    return render_scene


def build_shape(shape, label):
    material = Material(**shape._asdict())
    return Shape(label.transform, label.type, material, label.pattern)


def compute_loss(scene, shape, shape_label, true_image, render):
    shape = build_shape(shape, shape_label)
    pred_image, pred_depth = render(shape, scene)
    return jp.mean((true_image - pred_image) ** 2)


def initialize_landscape(key, optimizer, num_lights, pattern_shape):
    lights = []
    for arg in range(num_lights):
        key, subkey_0, subkey_1 = jax.random.split(key, 3)
        positions = jax.random.uniform(subkey_0, (3,), minval=-5.0, maxval=5.0)
        intensity = jax.random.uniform(subkey_1, (3,), minval=0.05, maxval=0.2)
        lights.append(PointLight(intensity, positions))
    floor = FLOOR(jp.array([0.5, 0.5, 0.5]), 0.1, 0.9, jp.zeros(pattern_shape))
    variable = LANDSCAPE_VARIABLE(floor, lights)
    return VARIABLE_STATE(optimizer.init(variable), variable)


def label_to_material_type(label):
    label = list(label.values())[0]
    material_type = "_".join([label["color"], label["material"]])
    return material_type


def update_state(optimizer, gradients, optimizer_state, variable):
    updates = optimizer.update(gradients, optimizer_state, variable)
    update, optimizer_state = updates
    variable = optax.apply_updates(variable, update)
    return optimizer_state, variable


def optimize_landscape(scene, material, label, image, optimizer):
    state, variable = scene.optimizer_state, scene.variable
    loss, grads = landscape_grad(variable, material.variable, label, image)
    state, variable = update_state(optimizer, grads, state, variable)
    return loss, state, variable


def parse_color(label):
    label = list(label.values())[0]
    return jp.array(label["RGB"]) / 255.0


def build_MATERIAL(optimizer, color, ambient, diffuse, specular, shininess):
    variable = MATERIAL_VARIABLE(color, ambient, diffuse, specular, shininess)
    optimizer_state = optimizer.init(variable)
    return VARIABLE_STATE(optimizer_state, variable)


def build_material_types(labels):
    material_types = {}
    for label in labels:
        material_type = label_to_material_type(label)
        material_types[material_type] = parse_color(label)
    return material_types


def initialize_materials(
    optimizer, labels, ambient, diffuse, specular, shininess
):
    material_types = build_material_types(labels)
    materials = {}
    for material_type, color in material_types.items():
        args = (optimizer, color, ambient, diffuse, specular, shininess)
        materials[material_type] = build_MATERIAL(*args)
    return materials


def label_to_shape(name_to_type, pattern_shape, label):
    label = list(label.values())[0]
    shift, theta, scale, color, shape_type = parse_label(label, name_to_type)
    shifts = paz.SE3.translation(jp.array([shift[0], scale[1], shift[1]]))
    rotate = paz.SE3.rotation_y(theta)
    scale = paz.SE3.scaling(jp.array(scale))
    transform = shifts @ rotate @ scale
    material = Material(jp.array(color), 0.1, 0.9, 0.5, 4.0)
    pattern = Pattern(jp.eye(4), NO_PATTERN, jp.zeros(pattern_shape))
    arg_type = jp.argmax(shape_type).tolist()
    return Shape(transform, ARG_TO_TYPE[arg_type], material, pattern)


def write_image(image, directory, filename):
    paz.image.write(Path(directory) / filename, paz.image.denormalize(image))


def write_images(render, labels, materials, landscape, directory):
    for label_arg, label in enumerate(labels):
        material = materials[label_to_material_type(label)]
        label = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
        shape = build_shape(material.variable, label)
        image, depth = render(shape, landscape.variable)
        write_image(image, directory, f"image_{label_arg:03d}.png")


key = jax.random.PRNGKey(args.seed)
root = Path(args.root) / args.dataset_name
root = paz.logger.make_timestamped_directory(root, args.label)
paz.logger.write_dictionary(args.__dict__, Path(root) / "parameters.json")

dataset_metadata = paz.datasets.fsclvr.parse_metadata(args.dataset_name)
camera_origin = jp.array(dataset_metadata["camera_origin"])
H, W = dataset_metadata["image_shape"]
y_FOV = dataset_metadata["y_FOV"]
image_shape = [int(H * args.viewport_factor), int(W * args.viewport_factor)]
dataset_path = Path(args.dataset_path) / args.dataset_name
dataset = paz.datasets.fsclvr.load(args.dataset_name, args.split, image_shape)
images, depths, labels = paz.datasets.fsclvr.flatten(dataset)

render = Scene(image_shape, y_FOV, camera_origin, args.shadows)
_compute_loss = paz.lock(compute_loss, render)
landscape_grad = jax.jit(jax.value_and_grad(_compute_loss, argnums=(0)))
materials_grad = jax.jit(jax.value_and_grad(_compute_loss, argnums=(1)))
landscape_optimizer = optax.adam(args.learning_rate)
materials_optimizer = optax.adam(args.learning_rate)

landscape_args = key, landscape_optimizer, args.num_lights, args.pattern_shape
landscape = initialize_landscape(*landscape_args)
materials = initialize_materials(
    materials_optimizer,
    labels,
    args.ambient,
    args.diffuse,
    args.specular,
    args.shininess,
)


images_directory = Path(root) / args.images_directory
landscape_loss = []
label, image = labels[0], images[0]
for inner_epoch in tqdm(range(args.inner_epochs)):
    material_type = label_to_material_type(label)
    material = materials[material_type]
    shape = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
    opt_args = (landscape, material, shape, image, landscape_optimizer)
    loss, state, landscape = optimize_landscape(*opt_args)
    landscape = VARIABLE_STATE(state, landscape)
    landscape_loss.append(loss)
    epoch_directory = f"epoch_{inner_epoch:02d}"
    epoch_directory = images_directory / epoch_directory
    paz.logger.make_directory(epoch_directory)
    write_images(
        jax.jit(render), [label], materials, landscape, epoch_directory
    )
