import argparse
import os
from collections import namedtuple
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
import jax.numpy as jp
import numpy as np
import optax
import trimesh

import paz
import paz.utils.plot as plot
from paz.backend import video
from paz.graphics.mesh import BinArgs, Mesh, build_sphere
from paz.graphics.mesh import compute_triangle_normals
from paz.graphics.mesh import count_binned_faces, tile_render_binned_soft_mask
from paz.graphics.types import Material, Pattern, PointLight
from paz.optimization.history import trim_trace

Parameters = namedtuple("Parameters", "offsets vertex_colors step_arg")
Views = namedtuple("Views", "images masks poses")
LossTerms = namedtuple("LossTerms", "rgb silhouette edge normal laplacian")


def optimize(initial, loss_fn, config):
    optimizer = build_optimizer()
    trace = []
    callbacks = [paz.optimization.TraceParameters(trace)]
    args = (initial, loss_fn, optimizer, config.max_steps)
    kwargs = {"callbacks": callbacks, "verbose": True}
    status, _, history = paz.minimize(*args, **kwargs)
    best_arg = int(jp.argmin(trim_trace(history).losses))
    return status, best_arg, trace, history


def build_optimizer():
    colors = optax.chain(optax.sgd(40.0, 0.9), clip_color_updates())
    transforms = {"offsets": optax.sgd(1.0, 0.9)}
    transforms["vertex_colors"] = colors
    transforms["step_arg"] = increment_step()
    labels = Parameters("offsets", "vertex_colors", "step_arg")
    return optax.multi_transform(transforms, labels)


def clip_color_updates():
    def project(updates, parameters):
        colors = parameters.vertex_colors
        clipped = jp.clip(colors + updates.vertex_colors, 0.0, 1.0) - colors
        return updates._replace(vertex_colors=clipped)

    return optax.stateless(project)


def increment_step():
    def project(updates, parameters):
        step_arg = jp.ones_like(parameters.step_arg)
        return updates._replace(step_arg=step_arg)

    return optax.stateless(project)


def build_view_schedule(config):
    views_per_step = 2
    key = jax.random.PRNGKey(config.seed)
    schedule = []
    for _ in range(config.max_steps):
        key, view_key = jax.random.split(key)
        view_args = jax.random.permutation(view_key, config.num_views)
        schedule.append(view_args[:views_per_step])
    return jp.stack(schedule)


def build_loss(sphere, views, face_pairs, degrees, view_schedule, config):
    def loss_fn(parameters):
        step_arg = jp.int32(jax.lax.stop_gradient(parameters.step_arg))
        step_views = subset_views(views, view_schedule[step_arg])
        args = (parameters, sphere, step_views, face_pairs, degrees, config)
        return weight_terms(compute_loss_terms(*args), config)

    return loss_fn


def subset_views(views, view_args):
    images = views.images[view_args]
    masks = views.masks[view_args]
    poses = views.poses[view_args]
    return Views(images, masks, poses)


def compute_loss_terms(parameters, sphere, views, face_pairs, degrees, config):
    mesh = build_predicted_mesh(parameters, sphere)
    rgb = compute_rgb_loss(mesh, views, config)
    silhouette = compute_silhouette_loss(mesh, views, config)
    edge = edge_length_loss(mesh.vertices, mesh.edges)
    face_args = jp.arange(len(mesh.faces))
    normals = compute_triangle_normals(mesh.vertices, mesh.faces, face_args)
    normal = normal_consistency_loss(normals, face_pairs)
    laplacian = laplacian_smoothing_loss(mesh.vertices, mesh.edges, degrees)
    return LossTerms(rgb, silhouette, edge, normal, laplacian)


def compute_rgb_loss(mesh, views, config):
    def step(total, view):
        pose, image = view
        prediction = render_mesh(mesh, pose, config)
        return total + paz.losses.mse(image, prediction), None

    total, _ = jax.lax.scan(step, jp.array(0.0), (views.poses, views.images))
    return total / views.poses.shape[0]


def compute_silhouette_loss(mesh, views, config):
    def step(total, view):
        pose, mask = view
        prediction = render_mesh_mask(mesh, pose, config)
        return total + paz.losses.mse(mask, prediction), None

    total, _ = jax.lax.scan(step, jp.array(0.0), (views.poses, views.masks))
    return total / views.poses.shape[0]


def weight_terms(terms, config):
    loss = config.rgb_weight * terms.rgb
    loss = loss + config.silhouette_weight * terms.silhouette
    loss = loss + config.edge_weight * terms.edge
    loss = loss + config.normal_weight * terms.normal
    loss = loss + config.laplacian_weight * terms.laplacian
    return loss


def edge_length_loss(vertices, edges):
    edge_vectors = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return jp.mean(jp.sum(edge_vectors * edge_vectors, axis=1))


def normal_consistency_loss(normals, face_pairs):
    normals_A = normals[face_pairs[:, 0]]
    normals_B = normals[face_pairs[:, 1]]
    return jp.mean(1.0 - jp.sum(normals_A * normals_B, axis=1))


def laplacian_smoothing_loss(vertices, edges, degrees):
    left, right = edges[:, 0], edges[:, 1]
    neighbors = jp.zeros_like(vertices)
    neighbors = neighbors.at[left].add(vertices[right])
    neighbors = neighbors.at[right].add(vertices[left])
    averages = neighbors / jp.maximum(degrees[:, None], 1.0)
    return jp.mean(jp.linalg.norm(averages - vertices, axis=1))


def load_cow_mesh(mesh_dir):
    mesh_dir = Path(mesh_dir)
    mesh = trimesh.load(mesh_dir / "cow.obj", force="mesh", process=False)
    vertices = jp.array(np.asarray(mesh.vertices), dtype=jp.float32)
    faces = jp.array(np.asarray(mesh.faces), dtype=jp.int32)
    vertices, center, scale = normalize_vertices(vertices)
    vertex_colors = jp.ones((len(vertices), 3))
    image = paz.image.normalize(paz.image.load(mesh_dir / "cow_texture.png"))
    pattern = Pattern(jp.eye(4), paz.graphics.NO_PATTERN, image)
    vertex_uvs = jp.array(np.asarray(mesh.visual.uv), dtype=jp.float32)
    args = (vertices, vertex_colors, jp.eye(4), build_material(), faces)
    args = args + (build_unique_edges(faces), pattern, vertex_uvs)
    return Mesh(*args), center, scale


def normalize_vertices(vertices):
    center = jp.mean(vertices, axis=0)
    scale = jp.max(jp.abs(vertices - center))
    return (vertices - center) / scale, center, scale


def build_sphere_mesh():
    vertices, faces, _ = build_sphere(1.0, 4)
    vertex_colors = jp.full(vertices.shape, 0.5)
    edges = build_unique_edges(faces)
    args = (vertices, vertex_colors, jp.eye(4), build_material(), faces, edges)
    return Mesh(*args)


def build_material():
    return Material(jp.zeros(3), 0.1, 0.9, 0.3, 64.0)


def build_initial_parameters(sphere):
    offsets = jp.zeros_like(sphere.vertices)
    step_arg = jp.array(0.0, dtype=jp.float32)
    return Parameters(offsets, sphere.vertex_colors, step_arg)


def build_predicted_mesh(parameters, sphere):
    vertices = sphere.vertices + parameters.offsets
    colors = parameters.vertex_colors
    return sphere._replace(vertices=vertices, vertex_colors=colors)


def build_unique_edges(faces):
    faces = paz.to_numpy(faces)
    edges_A = faces[:, [0, 1]]
    edges_B = faces[:, [1, 2]]
    edges_C = faces[:, [2, 0]]
    edges = np.sort(np.concatenate([edges_A, edges_B, edges_C]), axis=1)
    return jp.array(np.unique(edges, axis=0), dtype=jp.int32)


def build_face_pairs(sphere):
    args = (paz.to_numpy(sphere.vertices), paz.to_numpy(sphere.faces))
    mesh = trimesh.Trimesh(*args, process=False)
    return jp.array(mesh.face_adjacency, dtype=jp.int32)


def build_vertex_degrees(sphere):
    degrees = jp.zeros(len(sphere.vertices))
    degrees = degrees.at[sphere.edges[:, 0]].add(1.0)
    return degrees.at[sphere.edges[:, 1]].add(1.0)


def build_camera_poses(distance, num_views):
    elevations = jp.linspace(0.0, 360.0, num_views)
    azimuths = jp.linspace(-180.0, 180.0, num_views)
    origin_fn = jax.vmap(camera_origin, in_axes=(None, 0, 0))
    origins = origin_fn(distance, elevations, azimuths)
    return jax.vmap(camera_pose)(origins)


def camera_origin(distance, elevation, azimuth):
    elevation = jp.radians(elevation)
    azimuth = jp.radians(azimuth)
    x = distance * jp.cos(elevation) * jp.sin(azimuth)
    y = distance * jp.sin(elevation)
    z = distance * jp.cos(elevation) * jp.cos(azimuth)
    return jp.array([x, y, z])


def camera_pose(origin):
    target = jp.zeros(3)
    up = jp.array([0.0, 1.0, 0.0])
    forward = paz.algebra.normalize(target - origin)
    side_norm = jp.linalg.norm(jp.cross(forward, up))
    up = jp.where(side_norm < 1e-4, jp.array([0.0, 0.0, 1.0]), up)
    return paz.SE3.view_transform(origin, target, up)


def validate_binned_masks(cow, sphere, poses, config):
    cow_count = compute_max_bin_count(cow, poses, config)
    sphere_count = compute_max_bin_count(sphere, poses, config)
    max_count = max(cow_count, sphere_count)
    if max_count > build_bins(config).max_faces:
        message = "max_faces_per_bin must be at least "
        raise ValueError(message + str(max_count))


def compute_max_bin_count(mesh, poses, config):
    counts = [compute_bin_counts(mesh, pose, config) for pose in poses]
    return int(jp.max(jp.stack(counts)))


def compute_bin_counts(mesh, pose, config):
    shape = (config.image_size, config.image_size)
    args = (shape, pose, mesh, config.y_fov, config.mask_sigma)
    return count_binned_faces(*args, build_bins(config))


def build_bins(config):
    return BinArgs(config.mask_bin_size, config.max_faces_per_bin)


def render_target_views(mesh, poses, config):
    render_fn = jax.jit(lambda pose: render_mesh(mesh, pose, config))
    return jp.stack([render_fn(pose) for pose in poses])


def render_target_masks(mesh, poses, config):
    render_fn = jax.jit(lambda pose: render_mesh_mask(mesh, pose, config))
    return jp.stack([render_fn(pose) for pose in poses])


def render_mesh(mesh, pose, config):
    scene = paz.graphics.Scene([mesh])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, -3.0]))]
    shape = (config.image_size, config.image_size)
    args = (shape, config.y_fov, pose, scene, None, lights, (2, 2), 4096)
    image, _ = paz.graphics.render(*args)
    return image


def render_mesh_mask(mesh, pose, config):
    H = W = config.image_size
    args = (build_bins(config), config.y_fov, H, W, pose, mesh)
    args = args + (config.mask_sigma, 512)
    return tile_render_binned_soft_mask(*args)


def build_snapshots(trace, initial, best_arg):
    grid_every = 250
    snapshots = {0: initial}
    for step_arg in range(grid_every, len(trace), grid_every):
        snapshots[step_arg] = trace[step_arg]
    snapshots[best_arg] = trace[best_arg]
    return sorted(snapshots.items())


def compute_metrics(parameters, sphere, views, face_pairs, degrees, config):
    args = (parameters, sphere, views, face_pairs, degrees, config)
    terms = compute_loss_terms(*args)
    values = {name: float(term) for name, term in zip(terms._fields, terms)}
    values["total"] = float(weight_terms(terms, config))
    return values


def write_view_images(output_dir, initial, fitted, sphere, views, config):
    view_arg = select_view_arg(views)
    pose = views.poses[view_arg]
    initial_mesh = build_predicted_mesh(initial, sphere)
    fitted_mesh = build_predicted_mesh(fitted, sphere)
    images = [views.images[view_arg]]
    images.append(render_mesh(initial_mesh, pose, config))
    images.append(render_mesh(fitted_mesh, pose, config))
    masks = [views.masks[view_arg]]
    masks.append(render_mesh_mask(initial_mesh, pose, config))
    masks.append(render_mesh_mask(fitted_mesh, pose, config))
    write_image(output_dir / "comparison_view.png", jp.concatenate(images, 1))
    write_image(output_dir / "comparison_mask.png", jp.concatenate(masks, 1))


def write_trace_images(output_dir, snapshots, sphere, views, config):
    pose = views.poses[select_view_arg(views)]
    images, masks = [], []
    for _, parameters in snapshots:
        mesh = build_predicted_mesh(parameters, sphere)
        images.append(render_mesh(mesh, pose, config))
        masks.append(render_mesh_mask(mesh, pose, config))
    write_image(output_dir / "trace_grid.png", build_image_grid(images))
    write_image(output_dir / "trace_mask_grid.png", build_image_grid(masks))


def write_step_images(image_dir, trace, best_arg, sphere, views, config):
    for stale_path in image_dir.glob("step_*.png"):
        stale_path.unlink()
    pose = views.poses[select_view_arg(views)]
    render_fn = jax.jit(lambda mesh: render_mesh(mesh, pose, config))
    step_args = set(range(0, len(trace), config.save_every))
    image_paths = []
    for step_arg in sorted(step_args | {best_arg}):
        mesh = build_predicted_mesh(trace[step_arg], sphere)
        image_path = image_dir / f"step_{step_arg:05d}.png"
        write_image(image_path, render_fn(mesh))
        image_paths.append(str(image_path))
    return image_paths


def select_view_arg(views):
    return min(1, len(views.poses) - 1)


def build_image_grid(images):
    images = np.stack([paz.to_numpy(to_color(image)) for image in images])
    return paz.draw.mosaic(images, (-1, 5), background=1.0)


def to_color(image):
    if image.ndim == 2:
        image = jp.repeat(jp.expand_dims(image, -1), 3, axis=-1)
    return image


def write_image(path, image):
    image = jp.clip(to_color(image), 0.0, 1.0)
    paz.image.write(path, paz.image.denormalize(image))


def write_losses(output_dir, history):
    losses = paz.to_numpy(trim_trace(history).losses)
    np.savetxt(output_dir / "losses.csv", losses, delimiter=",")
    steps = list(range(1, len(losses) + 1))
    figure, axis = plot.subplots(1, 1, figsize=(8, 4))
    plot.line(steps, losses, axis=axis, color=plot.DEFAULT_PALETTE.primary)
    plot.clean(axis)
    plot.set_labels(axis, x="step", y="loss")
    plot.save(figure, str(output_dir / "losses.png"))


def write_obj(path, parameters, sphere, center, scale):
    vertices = sphere.vertices + parameters.offsets
    vertices = paz.to_numpy(vertices * scale + center)
    faces = paz.to_numpy(sphere.faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = to_rgba(parameters.vertex_colors)
    mesh.export(str(path))


def to_rgba(colors):
    colors = paz.to_numpy(jp.clip(colors, 0.0, 1.0))
    ones = np.ones((len(colors), 1))
    return (255.0 * np.concatenate([colors, ones], axis=1)).astype(np.uint8)


def write_summary(output_dir, config, history, status):
    losses = trim_trace(history).losses
    summary = vars(config).copy()
    summary["status"] = int(status)
    summary["stop_step"] = int(history.stop_step)
    summary["initial_loss"] = float(losses[0])
    summary["best_loss"] = float(jp.min(losses))
    summary["best_step"] = int(jp.argmin(losses)) + 1
    paz.file.write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="multi-view cow mesh fitting")
    add = parser.add_argument
    add("--mesh_dir", default=str(root / "data" / "cow_mesh"), type=str)
    add("--output_dir", default=str(root / "results"), type=str)
    add("--max_steps", default=2000, type=int)
    add("--seed", default=777, type=int)
    add("--image_size", default=128, type=int)
    add("--y_fov", default=jp.pi / 3.0, type=float)
    add("--num_views", default=20, type=int)
    add("--mask_sigma", default=1e-4, type=float)
    add("--mask_bin_size", default=16, type=int)
    add("--max_faces_per_bin", default=4608, type=int)
    add("--rgb_weight", default=1.0, type=float)
    add("--silhouette_weight", default=1.0, type=float)
    add("--edge_weight", default=1.0, type=float)
    add("--normal_weight", default=0.01, type=float)
    add("--laplacian_weight", default=1.0, type=float)
    add("--save_every", default=10, type=int)
    add("--video_fps", default=32, type=int)
    config = parser.parse_args()

    output_dir = Path(paz.directory.make(config.output_dir))
    cow, center, scale = load_cow_mesh(config.mesh_dir)
    sphere = build_sphere_mesh()
    poses = build_camera_poses(2.7, config.num_views)
    validate_binned_masks(cow, sphere, poses, config)
    images = render_target_views(cow, poses, config)
    masks = render_target_masks(cow, poses, config)
    views = Views(images, masks, poses)
    face_pairs = build_face_pairs(sphere)
    degrees = build_vertex_degrees(sphere)
    view_schedule = build_view_schedule(config)
    args = (sphere, views, face_pairs, degrees, view_schedule, config)
    loss_fn = build_loss(*args)
    initial = build_initial_parameters(sphere)
    status, best_arg, trace, history = optimize(initial, loss_fn, config)
    fitted = trace[best_arg]
    snapshots = build_snapshots(trace, initial, best_arg)
    write_view_images(output_dir, initial, fitted, sphere, views, config)
    write_trace_images(output_dir, snapshots, sphere, views, config)
    image_dir = Path(paz.directory.make(output_dir / "step_images"))
    args = (image_dir, trace, best_arg, sphere, views, config)
    image_paths = write_step_images(*args)
    video_name = str(output_dir / "optimization.mp4")
    video.from_paths(image_paths, video_name, config.video_fps)
    write_losses(output_dir, history)
    metrics_args = (sphere, views, face_pairs, degrees, config)
    metrics = {"initial": compute_metrics(initial, *metrics_args)}
    metrics["final"] = compute_metrics(fitted, *metrics_args)
    paz.file.write_json(metrics, output_dir / "metrics.json")
    write_obj(output_dir / "final_model.obj", fitted, sphere, center, scale)
    write_summary(output_dir, config, history, status)
