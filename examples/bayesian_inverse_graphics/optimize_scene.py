import argparse
import os
from pathlib import Path
from collections import namedtuple

parser = argparse.ArgumentParser(description="scene optimization")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--memory", default=0.90, type=float)
parser.add_argument("--device", default="gpu", type=str)
parser.add_argument("--dataset_name", default="plain", type=str)
parser.add_argument("--dataset_path", default="datasets", type=str)
parser.add_argument("--shapes_directory", default="shapes", type=str)
parser.add_argument("--images_directory", default="images", type=str)
parser.add_argument("--viewport_factor", default=1.0, type=float)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--split", default="train", type=str)
parser.add_argument("--label", default="SCENE-OPTIMIZATION", type=str)
parser.add_argument("--num_lights", default=5, type=int)
parser.add_argument("--pattern_shape", nargs="+", default=[200, 200, 3])
parser.add_argument("--outer_epochs", default=7, type=int)
parser.add_argument("--inner_epochs", default=5, type=int)
parser.add_argument("--shadow", default=True, type=bool)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--ambient", default=0.1, type=float)
parser.add_argument("--diffuse", default=0.9, type=float)
parser.add_argument("--specular", default=0.5, type=float)
parser.add_argument("--shininess", default=4.0, type=float)
args = parser.parse_args()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory)

import jax
import paz
import optax
import jax.numpy as jp
import matplotlib.pyplot as plt


def write_losses(losses, directory, filename):
    DANDELION = [0.992, 0.737, 0.258]
    figure, axis = plt.subplots()
    axis.plot(losses, color=DANDELION)
    axis.set_ylabel("loss")
    axis.set_xlabel("step")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fullpath = Path(directory) / filename
    figure.savefig(fullpath, bbox_inches="tight")
    plt.close()


def write_pytree_files(tree, directory, filename):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paz.pytree.to_pickle(tree, directory / f"{filename}.pkl")
    paz.graphics.save(directory / f"{filename}", tree)


def write_image(image, directory, filename):
    filepath = Path(directory) / filename
    paz.image.write(filepath, paz.image.denormalize(jp.clip(image, 0.0, 1.0)))


SHAPE_TYPES = [paz.graphics.SPHERE, paz.graphics.CUBE, paz.graphics.CYLINDER]
NAME_TO_TYPE = dict(zip(["sphere", "cube", "cylinder"], SHAPE_TYPES))
ARG_TO_TYPE = dict(zip([0, 1, 2], SHAPE_TYPES))
FLOOR = namedtuple("FLOOR", ["color", "ambient", "diffuse", "image"])
LANDSCAPE_VARIABLE = namedtuple("SCENE", ["floor", "lights"])
MATERIAL_FIELDS = ["color", "ambient", "diffuse", "specular", "shininess"]
MATERIAL_VARIABLE = namedtuple("MATERIAL_VARIABLE", MATERIAL_FIELDS)
VARIABLE_STATE = namedtuple("VARIABLE_STATE", ["optimizer_state", "variable"])


def build_render(camera_origin, size, y_FOV, shadows):
    origin = jp.array(camera_origin)
    target = jp.array([0.0, 0.0, 0.0])
    upward = jp.array([0.0, 1.0, 0.0])
    transform = paz.SE3.view_transform(origin, target, upward)
    rays = paz.graphics.camera.build_rays(size, y_FOV, transform)
    return paz.partial(
        paz.graphics.render,
        image_shape=size,
        world_to_camera=transform,
        rays=rays,
        shadows=shadows,
    )


def build_floor(floor):
    pattern_type = paz.graphics.PLANAR_PATTERN
    pattern = paz.graphics.Pattern(jp.eye(4), pattern_type, floor.image)
    material = (floor.color, floor.ambient, floor.diffuse, 0.0, 200.0)
    material = paz.graphics.Material(*material)
    return paz.graphics.Shape(jp.eye(4), paz.graphics.PLANE, material, pattern)


def build_shape(shape, label):
    material = paz.graphics.Material(**shape._asdict())
    shape = (label.transform, label.type, material, label.pattern)
    return paz.graphics.Shape(*shape)


def SceneLoss(render):
    def apply(scene, shape, shape_label, true_image):
        shape = build_shape(shape, shape_label)
        pred_image, pred_depth = render(shape, scene)
        return jp.mean((true_image - pred_image) ** 2)

    return apply


def Scene(size, y_FOV, camera_origin, shadows):
    render = build_render(camera_origin, size, y_FOV, shadows)

    def render_scene(shape, scene):
        floor = build_floor(scene.floor)
        shapes = paz.graphics.Scene([shape, floor])
        image, depth = render(scene=shapes, mask=None, lights=scene.lights)
        return jp.clip(image, 0.0, 1.0), depth

    return render_scene


def label_to_shape(name_to_type, pattern_shape, label):
    label = list(label.values())[0]
    label_data = paz.datasets.fsclvr.parse_label(label, name_to_type)
    shift, theta, scale, color, shape_type = label_data
    shifts = paz.SE3.translation(jp.array([shift[0], scale[1], shift[1]]))
    rotate = paz.SE3.rotation_y(theta)
    scale = paz.SE3.scaling(jp.array(scale))
    transform = shifts @ rotate @ scale
    material = paz.graphics.Material(jp.array(color), 0.1, 0.9, 0.5, 4.0)
    pattern = paz.graphics.Pattern()
    shape_type = ARG_TO_TYPE[jp.argmax(shape_type).tolist()]
    return paz.graphics.Shape(transform, shape_type, material, pattern)


def parse_color(label):
    label = list(label.values())[0]
    return jp.array(label["RGB"]) / 255.0


def label_to_material_type(label):
    label = list(label.values())[0]
    material_type = "_".join([label["color"], label["material"]])
    return material_type


def build_material_types(labels):
    material_types = {}
    for label in labels:
        material_type = label_to_material_type(label)
        material_types[material_type] = parse_color(label)
    return material_types


def build_MATERIAL(optimizer, color, ambient, diffuse, specular, shininess):
    variable = MATERIAL_VARIABLE(color, ambient, diffuse, specular, shininess)
    optimizer_state = optimizer.init(variable)
    return VARIABLE_STATE(optimizer_state, variable)


def initialize_materials(optimizer, labels, ambient, diffuse, specular, shiny):
    material_types = build_material_types(labels)
    materials = {}
    for material_type, color in material_types.items():
        args = (optimizer, color, ambient, diffuse, specular, shiny)
        materials[material_type] = build_MATERIAL(*args)
    return materials


def initialize_landscape(optimizer, key, num_lights, pattern_shape):
    lights = []
    for arg in range(num_lights):
        key, subkey_0, subkey_1 = jax.random.split(key, 3)
        positions = jax.random.uniform(subkey_0, (3,), minval=-5.0, maxval=5.0)
        intensity = jax.random.uniform(subkey_1, (3,), minval=0.10, maxval=0.50)
        lights.append(paz.graphics.PointLight(intensity, positions))
    floor = FLOOR(jp.array([0.5, 0.5, 0.5]), 0.1, 0.9, jp.zeros(pattern_shape))
    variable = LANDSCAPE_VARIABLE(floor, lights)
    optimizer_state = optimizer.init(variable)
    return VARIABLE_STATE(optimizer_state, variable)


def optimize_materials(scene, material, label, image, optimizer):
    state, variable = material.optimizer_state, material.variable
    loss, grads = materials_grad(scene.variable, variable, label, image)
    state, variable = update_state(optimizer, grads, state, variable)
    return loss, state, variable


def optimize_landscape(scene, material, label, image, optimizer):
    state, variable = scene.optimizer_state, scene.variable
    loss, grads = landscape_grad(variable, material.variable, label, image)
    state, variable = update_state(optimizer, grads, state, variable)
    return loss, state, variable


def update_state(optimizer, gradients, optimizer_state, variable):
    updates = optimizer.update(gradients, optimizer_state, variable)
    update, optimizer_state = updates
    variable = optax.apply_updates(variable, update)
    return optimizer_state, variable


def write_images(render, labels, materials, landscape, directory):
    for label_arg, label in enumerate(labels):
        material = materials[label_to_material_type(label)]
        label = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
        shape = build_shape(material.variable, label)
        image, depth = render(shape, landscape.variable)
        write_image(image, directory, f"image_{label_arg:03d}.png")


def leaf_to_list(leaf):
    return leaf.tolist() if isinstance(leaf, jp.ndarray) else leaf


def write_materials(material_variables, path):
    materials = {}
    for material_type in material_variables.keys():
        variable = material_variables[material_type].variable
        variable = jax.tree_util.tree_map(leaf_to_list, variable)
        variable = variable._asdict()
        materials[material_type] = variable
    write_pytree_files(materials, path, "materials")


key = jax.random.PRNGKey(args.seed)
root = Path(args.root) / args.dataset_name
root = paz.directory.make_timestamped(root, args.label)
paz.file.write_json(args.__dict__, Path(root) / "parameters.json")

dataset_metadata = paz.datasets.fsclvr.parse_metadata(args.dataset_name)
camera_origin = jp.array(dataset_metadata["camera_origin"])
H, W = dataset_metadata["image_shape"]
y_FOV = dataset_metadata["y_FOV"]
image_shape = [int(H * args.viewport_factor), int(W * args.viewport_factor)]
dataset = paz.datasets.fsclvr.load(args.dataset_name, args.split, image_shape)
images, depths, labels = paz.datasets.fsclvr.flatten(dataset)
render = Scene(image_shape, y_FOV, camera_origin, args.shadow)
scene_loss = SceneLoss(render)
landscape_grad = jax.jit(jax.value_and_grad(scene_loss, argnums=(0)))
materials_grad = jax.jit(jax.value_and_grad(scene_loss, argnums=(1)))
landscape_optimizer = optax.adam(args.learning_rate)
materials_optimizer = optax.adam(args.learning_rate)

landscape_args = landscape_optimizer, key, args.num_lights, args.pattern_shape
landscape = initialize_landscape(*landscape_args)
materials = initialize_materials(
    materials_optimizer,
    labels,
    args.ambient,
    args.diffuse,
    args.specular,
    args.shininess,
)

landscape_losses, materials_losses = [], {}
for material_type in materials.keys():
    materials_losses[material_type] = []

fast_render = jax.jit(render)
images_directory = Path(root) / args.images_directory
epoch_directory = paz.directory.make(str(images_directory / "epoch_00"))
write_images(fast_render, labels, materials, landscape, epoch_directory)
for outer_epoch in range(1, args.outer_epochs + 1):
    paz.message.info(f"Outer epoch {outer_epoch} / {args.outer_epochs}")
    for inner_epoch in range(args.inner_epochs):
        for label_arg, (label, image) in enumerate(zip(labels, images)):
            material_type = label_to_material_type(label)
            material = materials[material_type]
            shape = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
            opt_args = (landscape, material, shape, image, materials_optimizer)
            loss, state, variable = optimize_materials(*opt_args)
            materials[material_type] = VARIABLE_STATE(state, variable)
            materials_losses[material_type].append(loss)

    for inner_epoch in range(args.inner_epochs):
        for label_arg, (label, image) in enumerate(zip(labels, images)):
            material_type = label_to_material_type(label)
            material = materials[material_type]
            shape = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
            opt_args = (landscape, material, shape, image, landscape_optimizer)
            loss, state, landscape = optimize_landscape(*opt_args)
            landscape = VARIABLE_STATE(state, landscape)
            landscape_losses.append(loss)

    epoch_directory = f"epoch_{outer_epoch:02d}"
    epoch_directory = images_directory / epoch_directory
    paz.directory.make(epoch_directory)
    write_images(fast_render, labels, materials, landscape, epoch_directory)

write_pytree_files(landscape.variable.lights, root, "lights")
write_pytree_files(build_floor(landscape.variable.floor), root, "floor")

shapes_directory = paz.directory.make(Path(root) / args.shapes_directory)
for label_arg, label in enumerate(labels):
    material = materials[label_to_material_type(label)]
    label = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
    shape = build_shape(material.variable, label)
    write_pytree_files(shape, shapes_directory, f"shape_{label_arg:03d}")

write_materials(materials, root)

losses_directory = paz.directory.make(Path(root) / "losses")
write_pytree_files(landscape_losses, losses_directory, "landscape_losses")
write_losses(landscape_losses, losses_directory, "landscape_losses.pdf")
for material_type, losses in materials_losses.items():
    losses = jp.array(losses)
    write_pytree_files(losses, losses_directory, f"{material_type}_losses")
    write_losses(losses, losses_directory, f"{material_type}_losses.pdf")
