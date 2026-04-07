from pathlib import Path

import cv2
import jax.numpy as jp
import numpy as np
import paz
from paz.backend import video as video_utils
from paz.utils import plot
import trimesh

DANDELION = (1.0, 0.84, 0.0)


def build_cage(radius=1.0, subdivisions=4):
    mesh = trimesh.creation.icosphere(subdivisions, radius)
    vertices = jp.array(mesh.vertices.view(np.ndarray))
    faces = jp.array(mesh.faces.view(np.ndarray))
    return vertices, faces


def write_rgb_image(image, directory, filename):
    path = Path(directory) / filename
    image = np.array(paz.to_numpy(image))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (255.0 * image).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    paz.image.write(str(path), image)
    return str(path)


def write_frames(directory, frames, name_format="round_{:05d}.png"):
    for image_arg, image in enumerate(frames):
        name = name_format.format(image_arg)
        write_rgb_image(image, directory, name)


def write_mesh_obj(vertices, faces, vertex_colors, directory, filename):
    vertices = np.array(vertices)
    faces = np.array(faces)
    colors = np.array(vertex_colors)
    ones = np.ones((len(colors), 1))
    rgba = (255.0 * np.concatenate([colors, ones], axis=1)).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba)
    path = Path(directory) / filename
    mesh.export(str(path))


def draw_masks(image, masks, colors, alpha):
    image = np.array(image).copy()
    for mask, color in zip(masks, colors):
        mask_bool = np.array(mask).squeeze().astype(bool)
        overlay = np.array(color)[:3]
        image[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * overlay
    return image


def resize_masks(masks, target_size):
    H, W = target_size
    resized = []
    for mask in masks:
        mask_np = np.array(mask).squeeze()
        resized_mask = cv2.resize(mask_np, (W, H), cv2.INTER_LINEAR)
        resized.append(np.expand_dims(resized_mask > 0.5, axis=-1))
    return np.array(resized)


def plot_losses(losses, directory, filename, step_arg=None):
    figure, axis = plot.subplots()
    x_values = np.arange(len(losses))
    plot.line(x_values, np.array(losses), axis)
    if step_arg is not None:
        kwargs = {"x": step_arg, "ymax": 0.8, "color": "r"}
        axis.axvline(linestyle="dashed", label="best", **kwargs)
        plot.legend(axis)
    plot.set_labels(axis, "step", "loss")
    plot.clean(axis, "box")
    plot.save(figure, Path(directory) / filename)


def plot_metrics(metrics, directory, filename, step_arg=None):
    figure, axis = plot.subplots()
    color_map = plot.plt.get_cmap("tab10")
    metric_names = list(metrics.keys())
    colors = [color_map(i % 10) for i in range(len(metric_names))]
    for color, metric_name in zip(colors, metric_names):
        metric_values = metrics[metric_name]
        x_values = np.arange(len(metric_values))
        y_values = np.array(metric_values)
        line_args = (x_values, y_values, axis)
        plot.line(*line_args, color=color, label=metric_name)
    if step_arg is not None:
        axis.axvline(x=step_arg, ymax=0.8, color="r", linestyle="dashed")
    plot.legend(axis)
    plot.set_labels(axis, "step", "loss")
    plot.clean(axis, "box")
    plot.save(figure, Path(directory) / filename)


def export_orbit(root, round_path, video_name, fps, orbit, camera_origin):
    img_shape, y_FOV, scene, lights, shadows, num_views = orbit
    directory = paz.directory.make(Path(root) / round_path)
    args = (img_shape, y_FOV, scene, lights, shadows, camera_origin, num_views)
    frames = paz.graphics.render_orbit(*args)
    write_frames(directory, frames)
    video_path = Path(root) / f"{video_name}.mp4"
    video_utils.from_directory(directory, name=str(video_path), fps=fps)


def export_mesh_orbit(root, round_path, video_name, fps, orbit, camera_origin):
    meshes, image_shape, y_FOV, lights, num_views, chunk_size = orbit
    H, W = image_shape
    directory = paz.directory.make(Path(root) / round_path)
    mask = jp.ones(len(meshes.transform), dtype=bool)
    render_args = (meshes, mask, H, W, y_FOV, lights, chunk_size)
    render_frame = paz.graphics.mesh_renderer(*render_args)
    angles = jp.linspace(1.5 * jp.pi, 3.5 * jp.pi, num_views)
    frames = []
    for angle in angles:
        pose = paz.graphics.orbit_pose(camera_origin, angle)
        frames.append(render_frame(pose))
    write_frames(directory, frames)
    video_path = Path(root) / f"{video_name}.mp4"
    video_utils.from_directory(directory, name=str(video_path), fps=fps)
