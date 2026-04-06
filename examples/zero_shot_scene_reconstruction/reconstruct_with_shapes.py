import argparse
import os
import socket
import time
from pathlib import Path

from paz.backend.standard import str_to_bool


def build_parser():
    description = "Extract scene with diff-render"
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("--concept_arg", default=0, type=int)
    add("--gpu_memory", default=0.90, type=float)
    add("--gpu_device", default=0, type=int)
    add("--image_size_ratio", default=1.0, type=float)
    add("--compute_metrics", default=True, type=str_to_bool)
    add("--compute_image_callback", default=True, type=str_to_bool)
    add("--render_final_scene", default=True, type=str_to_bool)
    add("--shadows", default=False, type=str_to_bool)
    add("--min_depth", default=0.15, type=float)
    add("--max_depth", default=2.0, type=float)
    add("--scene_learning_rate", default=10.0, type=float)
    add("--shape_learning_rate", default=0.1, type=float)
    add("--material_shape_learning_rate", default=10.0, type=float)
    add("--max_line_steps", default=50, type=int)
    add("--line_search", default="armijo", choices=["armijo", "wolfe"])
    add("--scene_max_steps", default=150, type=int)
    add("--scene_tolerance", default=1e-2, type=float)
    add("--shape_max_steps", default=150, type=int)
    add("--shape_tolerance", default=1e-2, type=float)
    add("--LBFGS_memory_size", default=10, type=int)
    add("--scene_color_weight", default=2.0, type=float)
    add("--shape_color_weight", default=2.0, type=float)
    add("--depth_weight", default=30.0, type=float)
    add("--masks_weight", default=4.0, type=float)
    add("--material_weight", default=1.0, type=float)
    add("--scale_weight", default=0.0, type=float)
    add("--scale_priors", type=float, nargs="+", default=[0.3, 0.3, 0.3])
    add("--curvature", default=30.0, type=float)
    add("--ambient", default=0.1, type=float)
    add("--diffuse", default=0.1, type=float)
    add("--specular", default=0.1, type=float)
    add("--shininess", default=100.0, type=float)
    add("--num_lights", default=1, type=int)
    add("--max_intensity", default=0.5, type=float)
    add("--min_intensity", default=1.0, type=float)
    add("--ellipsoid_scaling", default=1e3, type=float)
    add("--ellipsoid_std_scale", default=2.0, type=float)
    add("--ellipsoid_x_scale", default=1.0, type=float)
    add("--image_scene_path", default="scene_trace", type=str)
    add("--image_material_path", default="material_trace", type=str)
    add("--image_shape_path", default="shape_trace", type=str)
    add("--round_path", default="round_images", type=str)
    add("--video_name", default="optimization", type=str)
    add("--scene_views", default=360, type=int)
    add("--video_FPS", default=32, type=int)
    add("--root", default="experiments", type=str)
    add("--label", default="optimized_shapes", type=str)
    add("--cache_compilation", default=True, type=str_to_bool)
    add("--cache_path", default="cache", type=str)
    add("--seed", default=777, type=int)
    return parser


def setup_environment(args):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.gpu_memory}"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_device}"


parser = build_parser()
args = parser.parse_args()
setup_environment(args)

import jax
import jax.numpy as jp
from jax import jit
import keras

import paz
from paz.datasets import clevrpose
from paz.backend import video as video_utils
from paz.optimization import TraceParameters
from paz.optimizers import LBFGS
from backend import ellipsoid
from backend.losses import (
    build_material_shape_loss,
    build_metrics,
    build_scene_loss,
    build_shape_loss,
)
from backend.model import (
    build_shape_model,
    parameters_to_scene,
    preprocess_observations,
)
from backend.scene import (
    build_camera_pose,
    initialize_floor_material,
    initialize_lines,
    initialize_lights,
    initialize_shape_materials,
)
from backend.utils import (
    export_orbit,
    plot_losses,
    plot_metrics,
    write_rgb_image,
)

if args.cache_compilation:
    from jax.experimental.compilation_cache import compilation_cache

    cache_path = paz.directory.make(Path(".") / args.cache_path)
    compilation_cache.set_cache_dir(cache_path)

label = f"{args.label}_{args.concept_arg:03d}"
dataset_root = Path(args.root) / "CLEVRPOSE"
root = paz.directory.make_timestamped(dataset_root, label)
paz.file.write_json(args.__dict__, Path(root) / "parameters.json")
keras.utils.set_random_seed(args.seed)
key = jax.random.PRNGKey(args.seed)
print("Loading CLEVRPOSE: single RGBD image as observations I, D (Sec. III)")
intrinsics = clevrpose.get_intrinsics()
y_FOV = clevrpose.get_y_FOV()
shot = clevrpose.load(args.concept_arg)[0]
pose_image = jp.array(shot.image)
true_image = jp.array(shot.image) / 255.0
true_depth = jp.array(shot.depth)
H, W, _ = true_image.shape
ratio = args.image_size_ratio
image_shape = [int(H * ratio), int(W * ratio)]
min_depth = args.min_depth
max_depth = args.max_depth

print("Projecting pointcloud C into camera frame using intrinsics K (Eq. 4)")
camera_args = (shot.pointcloud, intrinsics, (H, W), max_depth, args.seed)
camera_data = build_camera_pose(*camera_args)
world_to_camera_opengl = camera_data[0]
camera_origin = camera_data[1]
world_to_camera_opencv = camera_data[3]

print("Initializing lights and Phong materials from mean mask RGB (Sec. III.C)")
light_args = (key, args.num_lights, args.min_intensity, args.max_intensity)
lights = initialize_lights(*light_args)
shading = (args.ambient, args.diffuse, args.specular, args.shininess)
floor_material = initialize_floor_material(*shading)
shot_masks = shot.masks.copy()
masks = jp.array(shot_masks)
shape_materials = initialize_shape_materials(true_image, masks, shading)

print("Step 1: MAP ellipsoid estimation with Laplace/LogNormal priors (Eq. 11)")
fit_data = (args.seed, shot.pointcloud, shot.depth, shot_masks)
fit_config = (args.ellipsoid_scaling, args.ellipsoid_std_scale)
fit_camera = (intrinsics, max_depth, args.ellipsoid_x_scale)
fit_args = (*fit_data, *fit_config, *fit_camera)
start_time = time.perf_counter()
points3D, scale_vectors = ellipsoid.fit_scene(*fit_args)
total_time_1 = time.perf_counter() - start_time

print("Constraining positions to rays: p_k = t*d + o (line constraint, Fig. 3)")
directions3D, origins3D, distances = initialize_lines(camera_origin, points3D)
initial_scale = jax.vmap(paz.SE3.scaling)(scale_vectors)
print("Building differentiable JAX ray tracer with spherical shapes (Sec. IV)")
camera = (image_shape, y_FOV, world_to_camera_opengl, min_depth, max_depth)
geometry = (initial_scale, directions3D, origins3D)
model = build_shape_model(camera, geometry, args.shadows)
jit_model = jit(model)

obs_args = (true_image, true_depth, shot_masks, image_shape)
true_image, true_depth, true_masks = preprocess_observations(*obs_args)

background_mask = jp.logical_not(true_masks.sum(axis=0))

scene_parameters = [lights, floor_material]
materials = shape_materials
shapes = (jp.ones_like(scale_vectors), distances)


print("Stage 1: Scene loss on lights and floor (color + material, Eq. 14)")
scene_data = (model, true_image, background_mask, args.curvature)
scene_weights = [args.scene_color_weight, args.material_weight]
scene_loss = build_scene_loss(scene_weights, materials, shapes, scene_data)

scene_parameters_trace = None
if args.compute_image_callback:
    scene_parameters_trace = []
scene_callbacks = None
if scene_parameters_trace is not None:
    scene_callbacks = [TraceParameters(scene_parameters_trace)]
linesearch = paz.optimizers.LineSearch(args.max_line_steps, args.line_search)
opt_cfg = (args.LBFGS_memory_size, linesearch)

print("Running L-BFGS to optimize lights and floor material (Eq. 15)")
start_time = time.perf_counter()
learning_rate = args.scene_learning_rate
max_steps = args.scene_max_steps
tolerance = args.scene_tolerance
scene_opt = (scene_parameters, scene_loss, learning_rate, max_steps, tolerance)
output = LBFGS(*scene_opt, *opt_cfg, callbacks=scene_callbacks)
scene_parameters, history = output
total_time_2 = time.perf_counter() - start_time
lights, floor_material = scene_parameters
losses = history.losses
scene_state = (lights, floor_material)

image_scene_path = paz.directory.make(Path(root) / args.image_scene_path)
if args.compute_image_callback:
    for arg, scene_sample in enumerate(scene_parameters_trace):
        light_sample, floor_sample = scene_sample
        image_args = (light_sample, floor_sample, materials, *shapes)
        image = jit_model(*image_args)[0]
        image = jp.clip(image, 0.0, 1.0)
        write_rgb_image(image, image_scene_path, f"trace_{arg:05d}.png")

matrix_args = (intrinsics, world_to_camera_opencv)
camera_matrix = paz.pinhole.make_camera_matrix(*matrix_args)
pose_args = (geometry, floor_material, materials, shapes)
_, transforms, _, _ = parameters_to_scene(*pose_args)
image_pose = paz.draw.poses(pose_image, transforms, camera_matrix)
write_rgb_image(image_pose, root, "poses_1_colors_A.png")
plot_losses(losses, root, "scene_loss.pdf")
paz.pytree.to_pickle(losses, Path(root) / "scene_loss.pkl")

material_shape_parameters_trace = None
if args.compute_image_callback:
    material_shape_parameters_trace = []
material_callbacks = None
if material_shape_parameters_trace is not None:
    material_callbacks = [TraceParameters(material_shape_parameters_trace)]
print("Stage 2: Material loss on per-object Phong parameters (Eq. 14)")
material_data = (model, true_image, true_masks, args.curvature)
material_weights = [args.shape_color_weight, args.material_weight]
material_loss_args = (material_weights, scene_state, shapes, material_data)
material_loss = build_material_shape_loss(*material_loss_args)
print("Running L-BFGS to optimize per-object materials m_k (Eq. 15)")
start_time = time.perf_counter()
learning_rate = args.material_shape_learning_rate
max_steps = args.shape_max_steps
tolerance = args.shape_tolerance
materials_opt = (materials, material_loss, learning_rate, max_steps, tolerance)
output = LBFGS(*materials_opt, *opt_cfg, callbacks=material_callbacks)
materials, history = output
total_time_3 = time.perf_counter() - start_time
losses = history.losses

image_material_path = paz.directory.make(Path(root) / args.image_material_path)
if args.compute_image_callback:
    for arg, material_sample in enumerate(material_shape_parameters_trace):
        model_args = (lights, floor_material, material_sample, *shapes)
        image = jit_model(*model_args)[0]
        image = jp.clip(image, 0.0, 1.0)
        write_rgb_image(image, image_material_path, f"trace_{arg:05d}.png")

print("Stage 3: Shape loss on positions and scales (depth + mask, Eq. 14)")
shape_data = (model, true_depth, true_masks, args.scale_priors)
shape_weights = [args.depth_weight, args.masks_weight, args.scale_weight]
shape_loss = build_shape_loss(shape_weights, scene_state, materials, shape_data)

if args.compute_metrics:
    metric_weights = {
        "depth": args.depth_weight,
        "masks": args.masks_weight,
        "scale": args.scale_weight,
    }
    metrics_args = (metric_weights, scene_state, materials, shape_data)
    metrics = build_metrics(*metrics_args)
else:
    metrics = None

shape_parameters_trace = None
if args.compute_image_callback:
    shape_parameters_trace = []
shape_callbacks = None
if shape_parameters_trace is not None:
    shape_callbacks = [TraceParameters(shape_parameters_trace)]
print("Running L-BFGS to optimize positions p_k and scales s_k (Eq. 15)")
start_time = time.perf_counter()
learning_rate = args.shape_learning_rate
max_steps = args.shape_max_steps
tolerance = args.shape_tolerance
shapes_opt = (shapes, shape_loss, learning_rate, max_steps, tolerance)
shape_kwargs = {"metrics": metrics, "callbacks": shape_callbacks}
output = LBFGS(*shapes_opt, *opt_cfg, **shape_kwargs)
shapes, history = output
total_time_4 = time.perf_counter() - start_time
losses = history.losses
metrics_trace = history.metrics.trace

image_shape_path = paz.directory.make(Path(root) / args.image_shape_path)
if args.compute_image_callback:
    for arg, shape_sample in enumerate(shape_parameters_trace):
        model_args = (lights, floor_material, materials, *shape_sample)
        image = jit_model(*model_args)[0]
        image = jp.clip(image, 0.0, 1.0)
        write_rgb_image(image, image_shape_path, f"trace_{arg:05d}.png")

print("Rendering best scene R(theta) with final optimized parameters (Eq. 13)")
parameters = (lights, floor_material, materials, *shapes)
best_image, best_depth, best_masks, aux = jit_model(*parameters)
transforms = aux["transforms"]
image_pose = paz.draw.poses(pose_image, transforms, camera_matrix)
write_rgb_image(image_pose, root, "poses_2_colors_A.png")
plot_losses(losses, root, "shape_loss.pdf")
paz.pytree.to_pickle(losses, Path(root) / "shape_loss.pkl")

total_time = total_time_1 + total_time_2 + total_time_3 + total_time_4
meta_data = {
    "total_time": total_time,
    "time_1": total_time_1,
    "time_2": total_time_2,
    "time_3": total_time_3,
    "time_4": total_time_4,
    "num_parameters": paz.pytree.count_elements(parameters),
    "hostname": socket.gethostname(),
    "num_objects": len(true_masks),
}
paz.file.write_json(meta_data, Path(root) / "meta_data.json")

if args.compute_metrics:
    plot_metrics(metrics_trace, root, "metrics.pdf")
    paz.pytree.to_pickle(metrics_trace, Path(root) / "metrics.pkl")
    metrics_sum = jp.array(list(metrics_trace.values())).sum(axis=0)
    plot_losses(metrics_sum, root, "metrics_sum.pdf")

frame_rate = args.video_FPS
if args.compute_image_callback:
    video_name = os.path.join(root, f"{args.video_name}.mp4")
    paths = [image_scene_path, image_material_path, image_shape_path]
    video_utils.from_directories(paths, video_name=video_name, fps=frame_rate)

print("Assembling scene output: O_k, T_k, m_k, lights l (Eq. 1)")
pose_args = (geometry, floor_material, materials, shapes)
scene, transforms, translations, _ = parameters_to_scene(*pose_args)
paz.pytree.to_pickle(scene, Path(root) / "best_tamayo_scene.pkl")
paz.pytree.to_pickle(lights, Path(root) / "best_lights.pkl")
paz.pytree.to_pickle(scale_vectors, Path(root) / "scales3D.pkl")
paz.pytree.to_pickle(translations, Path(root) / "shift_vectors.pkl")
paz.pytree.to_pickle(parameters, Path(root) / "best_parameters.pkl")
write_rgb_image(best_image, root, "best_image.png")
best_depth_path = Path(root) / "best_depth.png"
paz.depth.write(jp.squeeze(best_depth), str(best_depth_path))
for shape_arg, mask in enumerate(best_masks):
    mask = jp.repeat(mask, 3, axis=-1)
    write_rgb_image(mask, root, f"best_mask_{shape_arg:03d}.png")

if args.render_final_scene:
    print("Rendering orbital video of the reconstructed scene (Fig. 6)")
    orbit = (image_shape, y_FOV, scene, lights, args.shadows, args.scene_views)
    round_name = f"{args.video_name}_round"
    round_path = args.round_path
    orbit_config = (root, round_path, round_name, frame_rate)
    export_orbit(*orbit_config, orbit, camera_origin)
    x, height, radius = jp.abs(camera_origin)
    ground_origin = jp.array([x, 0.10 * height, radius])
    ground_path = f"{round_path}_ground"
    ground_name = f"{args.video_name}_ground"
    orbit_config = (root, ground_path, ground_name, frame_rate)
    export_orbit(*orbit_config, orbit, ground_origin)
