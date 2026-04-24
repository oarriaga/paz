import argparse
import os
from collections import namedtuple
from pathlib import Path


def build_parser():
    root = Path(__file__).resolve().parent
    cow_dir = root / "data" / "cow_mesh"
    results_dir = root / "results"
    parser = argparse.ArgumentParser(description="multi-view cow mesh fitting")
    add = parser.add_argument
    add("--mesh_dir", default=str(cow_dir), type=str)
    add("--output_dir", default=str(results_dir), type=str)
    add("--image_size", default=128, type=int)
    add("--num_views", default=20, type=int)
    add("--views_per_step", default=2, type=int)
    add("--subdivisions", default=4, type=int)
    add("--max_steps", default=2000, type=int)
    add("--learning_rate", default=1.0, type=float)
    add("--color_learning_rate_scale", default=40.0, type=float)
    add("--momentum", default=0.9, type=float)
    add("--distance", default=2.7, type=float)
    add("--tile_shape", default=[2, 2], nargs=2, type=int)
    add("--chunk_size", default=256, type=int)
    add("--mask_bin_size", default=16, type=int)
    add("--max_faces_per_bin", default=4608, type=int)
    add("--soft_mask_sigma", default=1e-4, type=float)
    add("--save_every", default=250, type=int)
    add("--seed", default=777, type=int)
    add("--gpu_memory", default=0.90, type=float)
    add("--rgb_weight", default=1.0, type=float)
    add("--silhouette_weight", default=1.0, type=float)
    add("--edge_weight", default=1.0, type=float)
    add("--normal_weight", default=0.01, type=float)
    add("--laplacian_weight", default=1.0, type=float)
    return parser


config = build_parser().parse_args()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.gpu_memory)

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import optax
import paz
import trimesh
from paz.graphics.mesh import Mesh
from paz.graphics.types import Material
from paz.graphics.types import Pattern
from paz.graphics.types import PointLight

Parameters = namedtuple("Parameters", "offsets vertex_colors step_arg")
MeshArgs = namedtuple("MeshArgs", "vertices faces edges material transform")
RENDER = "image_shape y_fov tile chunk lights sigma bins"
RenderArgs = namedtuple("RenderArgs", RENDER)
TargetArgs = namedtuple("TargetArgs", "images masks poses")
CowArgs = namedtuple("CowArgs", "mesh center scale")
DepthRangeArgs = namedtuple("DepthRangeArgs", "min_depth max_depth")
LOSS = "render target source regularizers weights depth_range"
LossArgs = namedtuple("LossArgs", LOSS)
RegularizerArgs = namedtuple("RegularizerArgs", "edges face_pairs degrees")
WEIGHTS = "rgb silhouette edge normal laplacian"
Weights = namedtuple("Weights", WEIGHTS)
LossTerms = namedtuple("LossTerms", WEIGHTS)


def main(config):
    validate_config(config)
    output_dir = make_output_dir(config.output_dir)
    render_args = build_render_args(config)
    cow = load_cow_mesh(config.mesh_dir)
    args = (config.distance, config.num_views)
    poses = build_camera_poses(*args)
    source = build_source_mesh_args(config)
    initial = build_initial_parameters(source)
    validate_binned_masks(cow.mesh, source, initial, poses, render_args)
    images, depths = render_target_views(cow.mesh, poses, render_args)
    masks = render_target_masks(cow.mesh, poses, render_args)
    depth_range = compute_depth_range(depths)
    target = TargetArgs(images, masks, poses)
    regularizers = build_regularizers(source)
    weights = build_weights(config)
    args = (render_args, target, source, regularizers, weights, depth_range)
    loss_args = LossArgs(*args)
    fitted, history, status, trace = optimize(initial, loss_args, config)
    args = (output_dir, fitted, initial, history, loss_args, cow, trace)
    write_results(*args)
    write_run_config(output_dir, config, history, status, loss_args)


def validate_config(config):
    H_tiles, W_tiles = config.tile_shape
    if config.image_size % H_tiles != 0:
        raise ValueError("image_size must be divisible by tile_shape[0].")
    if config.image_size % W_tiles != 0:
        raise ValueError("image_size must be divisible by tile_shape[1].")
    if config.num_views < 1:
        raise ValueError("num_views must be positive.")
    if config.views_per_step < 1:
        raise ValueError("views_per_step must be positive.")
    if config.views_per_step > config.num_views:
        raise ValueError("views_per_step must not exceed num_views.")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if config.image_size % config.mask_bin_size != 0:
        raise ValueError("image_size must be divisible by mask_bin_size.")
    if config.max_faces_per_bin < 1:
        raise ValueError("max_faces_per_bin must be positive.")
    if config.save_every < 1:
        raise ValueError("save_every must be positive.")
    if config.color_learning_rate_scale <= 0.0:
        raise ValueError("color_learning_rate_scale must be positive.")


def optimize(initial, loss_args, config):
    view_schedule = build_view_schedule(config)
    loss_fn = build_loss(loss_args, view_schedule)
    optimizer = build_optimizer(config)
    parameters_trace = []
    callbacks = [paz.optimization.TraceParameters(parameters_trace)]
    args = (initial, loss_fn, optimizer, config.max_steps)
    kwargs = {"callbacks": callbacks, "verbose": True}
    status, final_parameters, history = paz.minimize(*args, **kwargs)
    best_arg = int(jax.device_get(jp.argmin(history.losses)))
    fitted = parameters_trace[best_arg]
    trace = build_trace(parameters_trace, final_parameters, history, config)
    return fitted, history, status, trace


def build_optimizer(config):
    color_rate = config.learning_rate * config.color_learning_rate_scale
    colors = optax.sgd(color_rate, config.momentum)
    colors = optax.chain(colors, clip_color_updates())
    transforms = {"offsets": optax.sgd(config.learning_rate, config.momentum)}
    transforms["vertex_colors"] = colors
    transforms["step_arg"] = increment_step()
    labels = Parameters("offsets", "vertex_colors", "step_arg")
    return optax.multi_transform(transforms, labels)


def clip_color_updates():
    def project(updates, parameters):
        colors = parameters.vertex_colors
        updates_now = updates.vertex_colors
        clipped = jp.clip(colors + updates_now, 0.0, 1.0) - colors
        return updates._replace(vertex_colors=clipped)

    return optax.stateless(project)


def increment_step():
    def project(updates, parameters):
        step_arg = jp.ones_like(parameters.step_arg)
        return updates._replace(step_arg=step_arg)

    return optax.stateless(project)


def build_view_schedule(config):
    key = jax.random.PRNGKey(config.seed)
    view_schedule = []
    for _ in range(config.max_steps):
        key, view_key = jax.random.split(key)
        view_schedule.append(sample_view_args(view_key, config))
    return jp.stack(view_schedule)


def build_loss(loss_args, view_schedule):
    def loss_fn(parameters):
        step_arg = jax.lax.stop_gradient(parameters.step_arg)
        step_arg = jp.int32(step_arg)
        target = subset_target(loss_args.target, view_schedule[step_arg])
        args = loss_args._replace(target=target)
        terms = compute_loss_terms(parameters, args)
        return weight_terms(terms, loss_args.weights)

    return loss_fn


def sample_view_args(key, config):
    view_args = jax.random.permutation(key, config.num_views)
    return view_args[: config.views_per_step]


def subset_target(target, view_args):
    images = target.images[view_args]
    masks = target.masks[view_args]
    poses = target.poses[view_args]
    return TargetArgs(images, masks, poses)


def build_trace(parameters_trace, final_parameters, history, config):
    stop_step = int(jax.device_get(history.stop_step))
    trace = []
    for step_arg in range(config.save_every, stop_step + 1, config.save_every):
        parameters = final_parameters
        if step_arg < stop_step:
            parameters = parameters_trace[step_arg]
        trace.append((step_arg, parameters))
    return trace


def compute_loss_terms(parameters, loss_args):
    rgb = compute_rgb_term(parameters, loss_args)
    silhouette = compute_silhouette_term(parameters, loss_args)
    vertices = loss_args.source.vertices + parameters.offsets
    edge = compute_edge_term(vertices, loss_args)
    normal = compute_normal_term(vertices, loss_args)
    laplacian = compute_laplacian_term(vertices, loss_args)
    return LossTerms(rgb, silhouette, edge, normal, laplacian)


def compute_rgb_term(parameters, loss_args):
    if loss_args.weights.rgb == 0.0:
        return zero_loss()
    mesh = build_source_mesh(parameters, loss_args.source)
    init = zero_loss()
    views = (loss_args.target.poses, loss_args.target.images)
    step = compute_rgb_view_step(mesh, loss_args)
    total, _ = jax.lax.scan(step, init, views)
    scale = 1.0 / loss_args.target.poses.shape[0]
    return total * scale


def compute_silhouette_term(parameters, loss_args):
    if loss_args.weights.silhouette == 0.0:
        return zero_loss()
    mesh = build_source_mesh(parameters, loss_args.source)
    init = zero_loss()
    views = (loss_args.target.poses, loss_args.target.masks)
    step = compute_silhouette_view_step(mesh, loss_args)
    total, _ = jax.lax.scan(step, init, views)
    scale = 1.0 / loss_args.target.poses.shape[0]
    return total * scale


def compute_rgb_view_step(mesh, loss_args):
    def step(carry, view):
        pose, target_image = view
        pred_image, _ = render_mesh(mesh, pose, loss_args.render)
        return carry + paz.losses.mse(target_image, pred_image), None

    return step


def compute_silhouette_view_step(mesh, loss_args):
    def step(carry, view):
        pose, target_mask = view
        pred_mask = render_mesh_mask(mesh, pose, loss_args.render)
        return carry + paz.losses.mse(target_mask, pred_mask), None

    return step


def compute_edge_term(vertices, loss_args):
    if loss_args.weights.edge == 0.0:
        return zero_loss()
    return edge_length_loss(vertices, loss_args.regularizers.edges)


def compute_normal_term(vertices, loss_args):
    if loss_args.weights.normal == 0.0:
        return zero_loss()
    normals = compute_face_normals(vertices, loss_args.source.faces)
    return normal_consistency_loss(normals, loss_args.regularizers.face_pairs)


def compute_laplacian_term(vertices, loss_args):
    if loss_args.weights.laplacian == 0.0:
        return zero_loss()
    regularizers = loss_args.regularizers
    args = (vertices, regularizers.edges, regularizers.degrees)
    return laplacian_smoothing_loss(*args)


def zero_loss():
    return jp.array(0.0)


def weight_terms(terms, weights):
    loss = weights.rgb * terms.rgb
    loss = loss + weights.silhouette * terms.silhouette
    loss = loss + weights.edge * terms.edge
    loss = loss + weights.normal * terms.normal
    loss = loss + weights.laplacian * terms.laplacian
    return loss


def build_render_args(config):
    image_shape = (config.image_size, config.image_size)
    tile_shape = tuple(config.tile_shape)
    bins = build_bin_args(config)
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, -3.0]))]
    args = (image_shape, jp.pi / 3.0, tile_shape, config.chunk_size, lights)
    args = args + (config.soft_mask_sigma, bins)
    return RenderArgs(*args)


def build_bin_args(config):
    max_faces = config.max_faces_per_bin
    return paz.graphics.mesh.BinArgs(config.mask_bin_size, max_faces)


def load_cow_mesh(mesh_dir):
    mesh_dir = Path(mesh_dir)
    mesh_path = mesh_dir / "cow.obj"
    texture_path = mesh_dir / "cow_texture.png"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = jp.array(np.asarray(mesh.vertices), dtype=jp.float32)
    faces = jp.array(np.asarray(mesh.faces), dtype=jp.int32)
    vertices, center, scale = normalize_target_vertices(vertices)
    edges = build_unique_edges(faces)
    vertex_colors = jp.ones((len(vertices), 3))
    image = paz.image.normalize(paz.image.load(texture_path))
    pattern = Pattern(jp.eye(4), paz.graphics.NO_PATTERN, image)
    vertex_uvs = jp.array(np.asarray(mesh.visual.uv), dtype=jp.float32)
    args = (vertices, vertex_colors, jp.eye(4), build_material())
    args = args + (faces, edges, pattern, vertex_uvs)
    return CowArgs(Mesh(*args), center, scale)


def normalize_target_vertices(vertices):
    center = jp.mean(vertices, axis=0)
    scale = jp.max(jp.abs(vertices - center))
    return (vertices - center) / scale, center, scale


def build_source_mesh_args(config):
    args = (1.0, config.subdivisions)
    vertices, faces, _ = paz.graphics.mesh.build_sphere(*args)
    edges = build_unique_edges(faces)
    return MeshArgs(vertices, faces, edges, build_material(), jp.eye(4))


def build_source_mesh(parameters, source):
    vertices = source.vertices + parameters.offsets
    args = (vertices, parameters.vertex_colors, source.transform)
    args = args + (source.material, source.faces, source.edges)
    return Mesh(*args)


def build_initial_parameters(source):
    offsets = jp.zeros_like(source.vertices)
    vertex_colors = jp.full(source.vertices.shape, 0.5)
    return Parameters(offsets, vertex_colors, jp.array(0.0, dtype=jp.float32))


def build_material():
    return Material(jp.zeros(3), 0.1, 0.9, 0.3, 64.0)


def build_weights(config):
    args = (config.rgb_weight, config.silhouette_weight, config.edge_weight)
    args = args + (config.normal_weight, config.laplacian_weight)
    return Weights(*args)


def validate_binned_masks(cow_mesh, source, initial, poses, render_args):
    source_mesh = build_source_mesh(initial, source)
    cow_count = compute_max_bin_count(cow_mesh, poses, render_args)
    source_count = compute_max_bin_count(source_mesh, poses, render_args)
    max_count = max(cow_count, source_count)
    if max_count <= render_args.bins.max_faces:
        return
    message = "max_faces_per_bin must be at least "
    raise ValueError(message + str(max_count))


def compute_max_bin_count(mesh, poses, render_args):
    counts = [compute_bin_counts(mesh, pose, render_args) for pose in poses]
    return int(jax.device_get(jp.max(jp.stack(counts))))


def compute_bin_counts(mesh, pose, render_args):
    args = (render_args.image_shape, pose, mesh, render_args.y_fov)
    args = args + (render_args.sigma, render_args.bins)
    return paz.graphics.mesh.count_binned_faces(*args)


def build_camera_poses(distance, num_views):
    elevations = jp.linspace(0.0, 360.0, num_views)
    azimuths = jp.linspace(-180.0, 180.0, num_views)
    origin_fn = jax.vmap(camera_origin, in_axes=(None, 0, 0))
    origins = origin_fn(distance, elevations, azimuths)
    return jax.vmap(camera_pose)(origins)


def camera_origin(distance, elevation, azimuth):
    elevation = elevation * jp.pi / 180.0
    azimuth = azimuth * jp.pi / 180.0
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


def render_target_views(mesh, poses, render_args):
    render_fn = jax.jit(lambda pose: render_mesh(mesh, pose, render_args))
    images, depths = [], []
    for pose in poses:
        image, depth = render_fn(pose)
        images.append(image)
        depths.append(depth)
    return jp.stack(images), jp.stack(depths)


def render_target_masks(mesh, poses, render_args):
    render_fn = jax.jit(lambda pose: render_mesh_mask(mesh, pose, render_args))
    return jp.stack([render_fn(pose) for pose in poses])


def render_mesh(mesh, pose, render_args):
    meshes, mask = paz.graphics.mesh.merge_meshes(mesh)
    H, W = render_args.image_shape
    args = (render_args.tile, render_args.y_fov, H, W, pose)
    args = args + (meshes, mask, render_args.lights, render_args.chunk)
    return paz.graphics.mesh.tile_render(*args)


def render_mesh_mask(mesh, pose, render_args):
    H, W = render_args.image_shape
    args = (render_args.bins, render_args.y_fov, H, W, pose, mesh)
    args = args + (render_args.sigma, render_args.chunk)
    return paz.graphics.mesh.tile_render_binned_soft_mask(*args)


def compute_depth_range(depths):
    valid = depths > 1e-5
    min_values = jp.where(valid, depths, jp.inf)
    max_values = jp.where(valid, depths, -jp.inf)
    min_depth = jp.min(min_values)
    max_depth = jp.max(max_values)
    margin = 0.05 * (max_depth - min_depth)
    return DepthRangeArgs(min_depth - margin, max_depth + margin)


def build_regularizers(source):
    degrees = build_vertex_degrees(len(source.vertices), source.edges)
    face_pairs = build_adjacent_face_pairs(source.faces)
    return RegularizerArgs(source.edges, face_pairs, degrees)


def edge_length_loss(vertices, edges):
    edge_vectors = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return jp.mean(jp.sum(edge_vectors * edge_vectors, axis=1))


def compute_face_normals(vertices, faces):
    points_A = vertices[faces[:, 0]]
    points_B = vertices[faces[:, 1]]
    points_C = vertices[faces[:, 2]]
    normals = jp.cross(points_B - points_A, points_C - points_A)
    return paz.algebra.normalize(normals)


def normal_consistency_loss(normals, face_pairs):
    if face_pairs.shape[0] == 0:
        return jp.array(0.0)
    normal_A = normals[face_pairs[:, 0]]
    normal_B = normals[face_pairs[:, 1]]
    cosine = jp.sum(normal_A * normal_B, axis=1)
    return jp.mean(1.0 - cosine)


def laplacian_smoothing_loss(vertices, edges, degrees):
    left, right = edges[:, 0], edges[:, 1]
    neighbors = jp.zeros_like(vertices)
    neighbors = neighbors.at[left].add(vertices[right])
    neighbors = neighbors.at[right].add(vertices[left])
    averages = neighbors / jp.maximum(degrees[:, None], 1.0)
    smoothed = averages - vertices
    return jp.mean(jp.linalg.norm(smoothed, axis=1))


def build_unique_edges(faces):
    faces = np.asarray(jax.device_get(faces), dtype=np.int32)
    edge_A = faces[:, [0, 1]]
    edge_B = faces[:, [1, 2]]
    edge_C = faces[:, [2, 0]]
    edges = np.concatenate([edge_A, edge_B, edge_C])
    edges = np.sort(edges, axis=1)
    return jp.array(np.unique(edges, axis=0), dtype=jp.int32)


def build_vertex_degrees(num_vertices, edges):
    edges = np.asarray(jax.device_get(edges), dtype=np.int32)
    degrees = np.zeros(num_vertices, dtype=np.float32)
    np.add.at(degrees, edges[:, 0], 1.0)
    np.add.at(degrees, edges[:, 1], 1.0)
    return jp.array(degrees)


def build_adjacent_face_pairs(faces):
    faces = np.asarray(jax.device_get(faces), dtype=np.int32)
    edge_to_faces = {}
    for face_arg, face in enumerate(faces):
        add_face_edges(edge_to_faces, face, face_arg)
    return jp.array(collect_face_pairs(edge_to_faces), dtype=jp.int32)


def add_face_edges(edge_to_faces, face, face_arg):
    for edge in face_edges(face):
        key = tuple(sorted(edge))
        edge_to_faces.setdefault(key, []).append(face_arg)


def face_edges(face):
    a, b, c = face
    return [(a, b), (b, c), (c, a)]


def collect_face_pairs(edge_to_faces):
    face_pairs = []
    for face_args in edge_to_faces.values():
        face_pairs.extend(pair_face_args(face_args))
    return face_pairs


def pair_face_args(face_args):
    pairs = []
    for left_arg in range(len(face_args)):
        for right_arg in range(left_arg + 1, len(face_args)):
            pairs.append((face_args[left_arg], face_args[right_arg]))
    return pairs


def write_results(output_dir, fitted, initial, history, loss_args, cow, trace):
    write_view_images(output_dir, fitted, initial, loss_args)
    write_trace_images(output_dir, fitted, initial, loss_args, trace, history)
    write_metrics(output_dir, fitted, initial, loss_args)
    write_losses(output_dir / "losses.png", history.losses)
    write_obj(output_dir / "final_model.obj", fitted, loss_args.source, cow)


def write_view_images(output_dir, fitted, initial, loss_args):
    view_arg = select_view_arg(loss_args.target.poses.shape[0])
    target_image = loss_args.target.images[view_arg]
    target_mask = loss_args.target.masks[view_arg]
    write_image(output_dir / "target_view.png", target_image)
    write_image(output_dir / "target_mask.png", target_mask)
    initial_image = render_parameters(initial, view_arg, loss_args)
    final_image = render_parameters(fitted, view_arg, loss_args)
    initial_mask = render_parameters_mask(initial, view_arg, loss_args)
    final_mask = render_parameters_mask(fitted, view_arg, loss_args)
    write_image(output_dir / "initial_view.png", initial_image)
    write_image(output_dir / "final_view.png", final_image)
    write_image(output_dir / "initial_mask.png", initial_mask)
    write_image(output_dir / "final_mask.png", final_mask)
    comparison = jp.concatenate([target_image, initial_image, final_image], 1)
    mask_comparison = jp.concatenate([target_mask, initial_mask, final_mask], 1)
    write_image(output_dir / "comparison_view.png", comparison)
    write_image(output_dir / "comparison_mask.png", mask_comparison)


def write_trace_images(output_dir, fitted, initial, loss_args, trace, history):
    trace_dir = make_output_dir(output_dir / "trace_images")
    best_step = int(jax.device_get(jp.argmin(history.losses) + 1))
    samples = [(0, initial)] + trace + [(best_step, fitted)]
    view_arg = select_view_arg(loss_args.target.poses.shape[0])
    images = []
    masks = []
    for step_arg, parameters in deduplicate_samples(samples):
        image = render_parameters(parameters, view_arg, loss_args)
        mask = render_parameters_mask(parameters, view_arg, loss_args)
        write_image(trace_dir / f"step_{step_arg:05d}.png", image)
        write_image(trace_dir / f"mask_step_{step_arg:05d}.png", mask)
        images.append(image)
        masks.append(mask)
    write_image(output_dir / "trace_grid.png", build_image_grid(images))
    write_image(output_dir / "trace_mask_grid.png", build_image_grid(masks))


def deduplicate_samples(samples):
    deduplicated = {}
    for step_arg, parameters in samples:
        deduplicated[step_arg] = parameters
    return sorted(deduplicated.items())


def build_image_grid(images):
    images = [jp.clip(image, 0.0, 1.0) for image in images]
    num_cols = min(5, len(images))
    num_rows = (len(images) + num_cols - 1) // num_cols
    blank = jp.ones_like(images[0])
    missing = num_rows * num_cols - len(images)
    images = images + [blank] * missing
    rows = []
    for row_arg in range(num_rows):
        start = row_arg * num_cols
        rows.append(jp.concatenate(images[start : start + num_cols], axis=1))
    return jp.concatenate(rows, axis=0)


def write_metrics(output_dir, fitted, initial, loss_args):
    metrics = {"initial": metrics_dict(initial, loss_args)}
    metrics["final"] = metrics_dict(fitted, loss_args)
    paz.file.write_json(metrics, output_dir / "metrics.json")


def metrics_dict(parameters, loss_args):
    terms = compute_loss_terms(parameters, loss_args)
    weighted = weight_terms(terms, loss_args.weights)
    values = {}
    for name in terms._fields:
        values[name] = term_to_float(getattr(terms, name))
    values["total"] = term_to_float(weighted)
    return values


def term_to_float(value):
    return float(jax.device_get(value))


def render_parameters(parameters, view_arg, loss_args):
    mesh = build_source_mesh(parameters, loss_args.source)
    pose = loss_args.target.poses[view_arg]
    image, _ = render_mesh(mesh, pose, loss_args.render)
    return image


def render_parameters_mask(parameters, view_arg, loss_args):
    mesh = build_source_mesh(parameters, loss_args.source)
    pose = loss_args.target.poses[view_arg]
    return render_mesh_mask(mesh, pose, loss_args.render)


def select_view_arg(num_views):
    return 1 if num_views > 1 else 0


def write_image(path, image):
    image = jp.clip(image, 0.0, 1.0)
    if image.ndim == 2:
        image = jp.repeat(jp.expand_dims(image, -1), 3, axis=-1)
    paz.image.write(path, paz.image.denormalize(image))


def write_losses(path, losses):
    values = np.asarray(jax.device_get(losses))
    np.savetxt(path.with_suffix(".csv"), values, delimiter=",")
    figure, axis = plt.subplots()
    axis.plot(values)
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_obj(path, parameters, source, cow):
    vertices = source.vertices + parameters.offsets
    vertices = vertices * cow.scale + cow.center
    faces = np.asarray(jax.device_get(source.faces))
    colors = jp.clip(parameters.vertex_colors, 0.0, 1.0)
    mesh = trimesh.Trimesh(vertices=to_numpy(vertices), faces=faces)
    mesh.visual.vertex_colors = to_rgba(colors)
    mesh.export(str(path))


def to_numpy(values):
    return np.asarray(jax.device_get(values))


def to_rgba(colors):
    colors = np.asarray(jax.device_get(colors))
    ones = np.ones((len(colors), 1))
    rgba = np.concatenate([colors, ones], axis=1)
    return (255.0 * rgba).astype(np.uint8)


def write_run_config(output_dir, config, history, status, loss_args):
    values = vars(config).copy()
    losses = history.losses
    best_loss = float(jax.device_get(jp.min(losses)))
    best_step = int(jax.device_get(jp.argmin(losses) + 1))
    values["status"] = int(jax.device_get(status))
    values["stop_step"] = int(jax.device_get(history.stop_step))
    values["initial_loss"] = float(jax.device_get(losses[0]))
    values["last_loss"] = float(jax.device_get(losses[-1]))
    values["final_loss"] = best_loss
    values["best_loss"] = best_loss
    values["best_step"] = best_step
    values["export_step"] = best_step
    values["min_depth"] = float(jax.device_get(loss_args.depth_range.min_depth))
    values["max_depth"] = float(jax.device_get(loss_args.depth_range.max_depth))
    paz.file.write_json(values, output_dir / "parameters.json")


def make_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main(config)
