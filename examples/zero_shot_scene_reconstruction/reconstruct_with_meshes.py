import argparse
import glob
import os
import pickle
import socket
import time
from pathlib import Path


def str_to_bool(value):
    if isinstance(value, bool):
        result = value
    else:
        value = value.lower()
        if value in {"true", "1", "yes", "y"}:
            result = True
        elif value in {"false", "0", "no", "n"}:
            result = False
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    return result


def build_parser():
    description = "Refine scene with mesh diff-render"
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("--concept_arg", default=0, type=int)
    add("--gpu_memory", default=0.90, type=float)
    add("--gpu_device", default=0, type=int)
    add("--image_size_ratio", default=0.20, type=float)
    add("--compute_image_callback", default=True, type=str_to_bool)
    add("--callback_frequency", default=10, type=int)
    add("--compute_metrics", default=True, type=str_to_bool)
    add("--render_final_scene", default=True, type=str_to_bool)
    add("--min_depth", default=0.15, type=float)
    add("--max_depth", default=2.0, type=float)
    add("--learning_rate", default=5e-3, type=float)
    add("--max_steps", default=5000, type=int)
    add("--early_stop_min_delta", default=1e-4, type=float)
    add("--early_stop_patience", default=3000, type=int)
    add("--mesh_subdivisions", default=2, type=int)
    add("--cage_radius", default=2.1, type=float)
    add("--cage_subdivisions", default=3, type=int)
    add("--floor_size", default=4.0, type=float)
    add("--tile_shape", type=int, nargs=2, default=[1, 1])
    add("--chunk_size", default=128, type=int)
    add("--color_weight", default=1.0, type=float)
    add("--depth_weight", default=100.0, type=float)
    add("--masks_weight", default=10.0, type=float)
    add("--smooth_weight", default=1e-4, type=float)
    add("--volume_weight", default=1.0, type=float)
    add("--smooth_depth_weight", default=1.0, type=float)
    add("--curvature", default=30.0, type=float)
    add("--shapes_label", default="optimized_shapes", type=str)
    add("--image_mesh_path", default="mesh_trace", type=str)
    add("--round_path", default="round_images", type=str)
    add("--video_name", default="mesh_optimization", type=str)
    add("--scene_views", default=360, type=int)
    add("--video_FPS", default=16, type=int)
    add("--root", default="experiments", type=str)
    add("--label", default="optimized_meshes", type=str)
    add("--cache_compilation", default=True, type=str_to_bool)
    add("--cache_path", default="cache", type=str)
    add("--seed", default=777, type=int)
    return parser


def setup_environment(args):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.gpu_memory}"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_device}"


def build_callback_args(best_step_arg, frequency):
    callback_args = list(range(0, best_step_arg + 1, frequency))
    if callback_args[-1] != best_step_arg:
        callback_args.append(best_step_arg)
    return callback_args


parser = build_parser()
args = parser.parse_args()
setup_environment(args)

import jax
import jax.numpy as jp
from jax import jit
import keras
import optax

import paz
from paz.backend.mesh import build_laplacian
from paz.datasets import clevrpose
from paz.backend import video as video_utils
from paz.optimization import MAX_STEPS_REACHED
from paz.optimization import TraceParameters
from backend.losses import build_mesh_loss, build_mesh_metrics
from backend.mesh import (
    append_floor,
    build_floor,
    build_object_meshes,
)
from backend.model import build_mesh_model, preprocess_observations
from backend.scene import build_camera_pose
from backend.utils import (
    DANDELION,
    build_cage,
    draw_masks,
    export_mesh_orbit,
    plot_losses,
    plot_metrics,
    resize_masks,
    write_mesh_obj,
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

print("Loading CLEVRPOSE: single RGBD image as observations I, D (Sec. III)")
intrinsics = clevrpose.get_intrinsics()
y_FOV = clevrpose.get_y_FOV()
shot = clevrpose.load(args.concept_arg)[0]
true_image = jp.array(shot.image) / 255.0
true_depth = jp.array(shot.depth)
H, W, _ = true_image.shape
ratio = args.image_size_ratio
image_shape = [int(H * ratio), int(W * ratio)]
min_depth, max_depth = args.min_depth, args.max_depth

print("Projecting pointcloud C into camera frame using intrinsics K (Eq. 4)")
camera_args = (shot.pointcloud, intrinsics, (H, W), max_depth, args.seed)
camera_data = build_camera_pose(*camera_args)
world_to_camera_opengl = camera_data[0]
camera_origin = camera_data[1]
world_to_camera_opencv = camera_data[3]

print("Loading shape optimization results (Sec. III.C)")
shapes_card = f"*{args.shapes_label}_{args.concept_arg:03d}"
shapes_glob = str(dataset_root / shapes_card)
shapes_matches = sorted(glob.glob(shapes_glob))
if not shapes_matches:
    raise FileNotFoundError(f"No shapes results found: {shapes_glob}")
shapes_root = Path(shapes_matches[-1])
params_path = shapes_root / "best_parameters.pkl"
shape_parameters = pickle.load(open(params_path, "rb"))
lights, floor_material, materials, delta_scales, distances = shape_parameters
scale_vectors = pickle.load(open(shapes_root / "scales3D.pkl", "rb"))
shift_vectors = pickle.load(open(shapes_root / "shift_vectors.pkl", "rb"))

print("Reconstructing object transforms from shape optimization results")
shifts = jax.vmap(paz.SE3.translation)(shift_vectors)
start_scales = jax.vmap(paz.SE3.scaling)(scale_vectors)
delta_scale_mats = jax.vmap(paz.SE3.scaling)(delta_scales)
batchmul = jax.vmap(jp.matmul)
transforms = batchmul(shifts, batchmul(delta_scale_mats, start_scales))
num_objects = len(transforms)

print("Building sphere meshes and cage for deformation (Eq. 18)")
meshes = build_object_meshes(args.mesh_subdivisions, materials, transforms)
cage_verts, cage_faces = build_cage(args.cage_radius, args.cage_subdivisions)
sphere_verts = paz.graphics.mesh.build_sphere(1.0, args.mesh_subdivisions)[0]
mesh_weights = paz.cage.compute_mesh_weights(
    sphere_verts, cage_verts, cage_faces
)

print("Building floor mesh and Laplacian regularizer")
floor = build_floor(floor_material, args.floor_size, args.mesh_subdivisions)
laplacian = build_laplacian(sphere_verts, meshes.faces[0])
initial_volumes = jax.vmap(paz.mesh.compute_volume)(
    meshes.vertices, meshes.faces
)

print("Building differentiable JAX mesh ray tracer (Sec. IV)")
obs_args = (true_image, true_depth, shot.masks.copy(), image_shape)
true_image, true_depth, true_masks = preprocess_observations(*obs_args)
camera = (image_shape, y_FOV, world_to_camera_opengl, min_depth, max_depth)
camera = (*camera, args.tile_shape, args.chunk_size)
model = build_mesh_model(camera, meshes, mesh_weights, floor, lights)
jit_model = jit(model)

print("Building mesh loss (Eq. 17)")
loss_weights = [
    args.color_weight,
    args.depth_weight,
    args.masks_weight,
    args.smooth_weight,
    args.volume_weight,
    args.smooth_depth_weight,
]
metric_weights = {
    "image": args.color_weight,
    "depth": args.depth_weight,
    "masks": args.masks_weight,
    "smooth_mesh": args.smooth_weight,
    "volume": args.volume_weight,
    "smooth_depth": args.smooth_depth_weight,
}
observations = (true_image, true_depth, true_masks)
reg_data = (laplacian, initial_volumes, args.curvature)
mesh_loss = build_mesh_loss(loss_weights, model, observations, reg_data)
mesh_metrics = None
if args.compute_metrics:
    metric_args = (metric_weights, model, observations, reg_data)
    mesh_metrics = build_mesh_metrics(*metric_args)

cage_vertices = jp.tile(cage_verts, (num_objects, 1, 1))
optimizer = optax.adamw(args.learning_rate)
patience_args = (args.early_stop_min_delta, args.early_stop_patience)
stop_fn = paz.optimization.patience_stop(*patience_args)
mesh_trace = []
callbacks = [TraceParameters(mesh_trace)]

print("Running ADAMW to optimize cage vertices (Eq. 18)")
start_time = time.perf_counter()
opt_args = (cage_vertices, mesh_loss, optimizer, args.max_steps)
opt_kwargs = dict(stop_fn=stop_fn, metrics=mesh_metrics)
opt_kwargs.update(callbacks=callbacks, verbose=True)
status, cage_vertices, history = paz.minimize(*opt_args, **opt_kwargs)
total_time = time.perf_counter() - start_time
history = paz.optimization.trim_trace(history)
losses = history.losses
metrics_trace = history.metrics.trace
if status == MAX_STEPS_REACHED:
    losses = jp.concatenate([losses, jp.array([mesh_loss(cage_vertices)])])
    mesh_trace.append(cage_vertices)
    if args.compute_metrics:
        final_metrics = mesh_metrics(cage_vertices)
        append = lambda trace, value: jp.concatenate([trace, value[None]])
        metrics_trace = jax.tree.map(append, metrics_trace, final_metrics)
best_step_arg = int(jax.device_get(jp.argmin(losses)))
best_cage_vertices = mesh_trace[best_step_arg]
last_cage_vertices = cage_vertices

if args.compute_image_callback:
    image_mesh_path = paz.directory.make(Path(root) / args.image_mesh_path)
    callback_args = build_callback_args(best_step_arg, args.callback_frequency)
    for arg in callback_args:
        sample = mesh_trace[arg]
        image = jit_model(sample)[0]
        image = jp.clip(image, 0.0, 1.0)
        write_rgb_image(image, image_mesh_path, f"trace_{arg:05d}.png")

print("Rendering best mesh scene with final optimized parameters (Eq. 13)")
state_map = {
    "best": best_cage_vertices,
    "last": last_cage_vertices,
}
mesh_states, scene_states = {}, {}
for prefix, state in state_map.items():
    image, depth, masks, aux = jit_model(state)
    meshes_now = aux["meshes"]
    scene_now = append_floor(meshes_now, floor)
    mesh_states[prefix] = (image, depth, masks, meshes_now)
    scene_states[prefix] = scene_now
    write_rgb_image(image, root, f"{prefix}_image.png")
    depth_path = Path(root) / f"{prefix}_depth.png"
    paz.depth.write(jp.squeeze(depth), str(depth_path))
    for mask_arg, mask in enumerate(masks):
        mask = jp.repeat(mask, 3, axis=-1)
        name = f"{prefix}_mask_{mask_arg:03d}.png"
        write_rgb_image(mask, root, name)
    paz.pytree.to_pickle(state, Path(root) / f"{prefix}_parameters.pkl")
    paz.pytree.to_pickle(state, Path(root) / f"{prefix}_cage_vertices.pkl")
    paz.pytree.to_pickle(meshes_now, Path(root) / f"{prefix}_meshes.pkl")
    scene_path = Path(root) / f"{prefix}_tamayo_scene.pkl"
    paz.pytree.to_pickle(scene_now, scene_path)
    for mesh_arg in range(num_objects):
        vertices = meshes_now.vertices[mesh_arg]
        faces = meshes_now.faces[mesh_arg]
        colors = meshes_now.vertex_colors[mesh_arg]
        name = f"{prefix}_mesh_{mesh_arg:03d}.obj"
        write_mesh_obj(vertices, faces, colors, root, name)

best_image, best_depth, best_masks, best_meshes = mesh_states["best"]
best_scene = scene_states["best"]

plot_losses(losses, root, "mesh_loss.pdf", best_step_arg)
paz.pytree.to_pickle(losses, Path(root) / "mesh_loss.pkl")
paz.pytree.to_pickle(best_cage_vertices, Path(root) / "best_parameters.pkl")
paz.pytree.to_pickle(lights, Path(root) / "best_lights.pkl")
if args.compute_metrics:
    plot_metrics(metrics_trace, root, "metrics.pdf", best_step_arg)
    paz.pytree.to_pickle(metrics_trace, Path(root) / "metrics.pkl")

meta_data = {
    "total_time": total_time,
    "num_parameters": paz.pytree.count_elements(best_cage_vertices),
    "hostname": socket.gethostname(),
    "num_objects": num_objects,
    "best_step": best_step_arg,
}
paz.file.write_json(meta_data, Path(root) / "meta_data.json")

print("Drawing pose visualizations (color scheme A and B)")
colors_A = paz.draw.lincolor(num_objects, normalize=True)
colors_B = [DANDELION] * num_objects
pose_img = jp.array(shot.image) / 255.0
matrix_args = (intrinsics, world_to_camera_opencv)
K = paz.pinhole.make_camera_matrix(*matrix_args)
for scheme, colors in [("A", colors_A), ("B", colors_B)]:
    for with_mesh in [False, True]:
        suffix = "_and_mesh" if with_mesh else ""
        name = f"poses_color_{scheme}{suffix}.png"
        img = pose_img.copy()
        img = paz.draw.poses(img, best_meshes.transform, K, 2, 4, colors)
        if with_mesh:
            draw_args = img, best_meshes, K, 2, 4, colors
            img = paz.draw.mesh_poses(*draw_args, edge_scale=0.6)
        write_rgb_image(img, root, name)

print("Drawing mask overlays with bounding boxes")
pred_masks_resized = resize_masks(best_masks, (H, W))
pred_boxes, valid_args = [], []
for arg, mask in enumerate(pred_masks_resized):
    box = paz.mask.to_box(paz.to_numpy(mask).squeeze().astype(float), 1.0)
    box = paz.to_numpy(box).astype(int)
    if box[2] >= box[0] and box[3] >= box[1]:
        pred_boxes.append(box)
        valid_args.append(arg)
for scheme, colors in [("A", colors_A), ("B", colors_B)]:
    img = draw_masks(pose_img.copy(), pred_masks_resized, colors, 0.6)
    write_rgb_image(img, root, f"masks_color_{scheme}.png")
    valid_colors = [colors[arg] for arg in valid_args]
    if len(pred_boxes) > 0:
        img = paz.draw.boxes(img, pred_boxes, valid_colors, 2)
    write_rgb_image(img, root, f"masks_boxes_color_{scheme}.png")

frame_rate = args.video_FPS
if args.compute_image_callback:
    video_name = os.path.join(root, f"{args.video_name}.mp4")
    video_args = (image_mesh_path,)
    video_utils.from_directory(*video_args, name=video_name, fps=frame_rate)

if args.render_final_scene:
    print("Rendering orbital video of the reconstructed mesh scene")
    orbit = (best_scene, image_shape, y_FOV, lights, args.scene_views, 128)
    video_name = f"{args.video_name}_round"
    path_args = (root, args.round_path, video_name, frame_rate)
    export_mesh_orbit(*path_args, orbit, camera_origin)
    x, height, radius = jp.abs(camera_origin)
    ground_origin = jp.array([x, 0.10 * height, radius])
    ground_path = f"{args.round_path}_ground"
    video_name = f"{args.video_name}_ground"
    path_args = (root, ground_path, video_name, frame_rate)
    export_mesh_orbit(*path_args, orbit, ground_origin)
