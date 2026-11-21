import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
from keras.utils import get_file

import paz
from paz.graphics import AreaLight, Material, Mesh, Scene
from paz.graphics.mesh import compute_vertex_normals
import figures


def write_material_figures(key, materials, directory, size, scale, light_steps):
    vertices, groups = load_shader_ball()
    render_args = (vertices, groups, size * scale, light_steps)
    render = jax.jit(paz.lock(render_material, *render_args))
    panels = {}
    for name in sorted(materials, key=sort_by_material):
        image, _ = render(key, build_material(materials[name]))
        image = jp.clip(image, 0.0, 1.0)
        image = paz.image.resize(image, (size, size), "linear", True)
        filepath = directory / f"material_{name}.png"
        paz.image.write(str(filepath), paz.image.denormalize(image))
        panels[name] = (figures.draw_image, image, name.replace("_", " "))
    figures.save(figures.plot_sheet(panels, 4, 1.8), directory, "materials")


def build_material(properties):
    arrays = {name: jp.array(value) for name, value in properties.items()}
    return Material(**arrays)


def sort_by_material(name):
    color, material = name.split("_")
    return material, color


def load_shader_ball():
    # Public domain shader ball by Mat Makin (github.com/derkreature).
    URL = "https://raw.githubusercontent.com/derkreature/ShaderBall/"
    URL = URL + "master/shaderBallNoCrease/shaderBall.obj"
    cache = "paz/meshes/shader_ball_makin"
    filepath = get_file("shaderBall.obj", URL, cache_subdir=cache)
    vertices, groups = load_groups(filepath)
    return normalize_vertices(vertices), groups


def load_groups(filepath):
    # trimesh.load fails on OBJ files with the installed numpy.
    vertices, groups, name = [], {}, None
    for line in open(filepath, "r"):
        values = line.split()
        if line.startswith("v "):
            vertices.append([float(value) for value in values[1:4]])
        elif line.startswith("g "):
            name = values[1]
        elif line.startswith("f "):
            corners = [int(value.split("/")[0]) - 1 for value in values[1:]]
            groups.setdefault(name, []).extend(triangulate(corners))
    groups = {name: jp.array(faces) for name, faces in groups.items()}
    return jp.array(vertices), groups


def triangulate(corners):
    triangles = []
    for arg in range(1, len(corners) - 1):
        triangles.append([corners[0], corners[arg], corners[arg + 1]])
    return triangles


def normalize_vertices(vertices):
    lower, upper = jp.min(vertices, axis=0), jp.max(vertices, axis=0)
    center = (lower + upper) / 2.0
    floor_center = center.at[1].set(lower[1])
    return (vertices - floor_center) / (upper[1] - lower[1])


def render_material(key, material, vertices, groups, size, light_steps):
    shape = (size, size)
    intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 6.5, *shape)
    camera_origin = jp.array([1.3, 1.5, 3.2])
    camera_target = jp.array([0.08, 0.45, 0.0])
    upward = jp.array([0.0, 1.0, 0.0])
    pose = paz.SE3.view_transform(camera_origin, camera_target, upward)
    floor_material = Material(jp.full(3, 0.28), 0.45, 0.55, 0.0, 100.0, 0.1)
    floor = paz.graphics.Plane(material=floor_material)
    nodes = [*build_shader_ball(vertices, groups, material), floor]
    kwargs = dict(scene=Scene(nodes), lights=build_lights(key, light_steps))
    kwargs.update(mask=None, tiles=(4, 4), chunk_size=8192, shadows=True)
    kwargs.update(num_bounces=2, face_chunk_size=1024)
    return paz.graphics.render(shape, intrinsics, pose, **kwargs)


def build_lights(key, num_steps):
    # The front and rim lights reveal the broad specular lobe of the metals.
    keys = jax.random.split(key, 3)
    main = build_area_light(keys[0], 0.65, [-1.6, 5.7, 3.3], num_steps)
    front = build_area_light(keys[1], 0.5, [2.0, 3.2, 5.0], num_steps // 3)
    rim = build_area_light(keys[2], 0.5, [-3.0, 3.0, -3.0], num_steps // 3)
    return [main, front, rim]


def build_area_light(key, intensity, center, num_steps, half_size=0.5):
    center = jp.array(center)
    forward = -center / jp.linalg.norm(center)
    rightward = jp.cross(forward, jp.array([0.0, 1.0, 0.0]))
    rightward = rightward / jp.linalg.norm(rightward)
    upward = jp.cross(rightward, forward)
    corner = center - half_size * (rightward + upward)
    edges = (2.0 * half_size * rightward, 2.0 * half_size * upward)
    steps = (num_steps, num_steps, key)
    return AreaLight(jp.full(3, intensity), corner, *edges, *steps)


def build_shader_ball(vertices, groups, material):
    neutral = Material(jp.full(3, 0.35), 0.3, 0.7, 0.1, 20.0)
    meshes = []
    for name, faces in groups.items():
        is_material_part = name.startswith("Material")
        part_material = material if is_material_part else neutral
        meshes.append(build_mesh(vertices, faces, part_material))
    return meshes


def build_mesh(vertices, faces, material):
    colors = jp.tile(material.color, (len(vertices), 1))
    pairs = [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    edges = jp.concatenate(pairs, axis=0)
    normals = compute_vertex_normals(vertices, faces)
    args = (vertices, colors, jp.eye(4), material, faces, edges)
    return Mesh(*args, vertex_normals=normals)


if __name__ == "__main__":
    description = "Draw the optimized materials on a shader ball"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--supersampling", default=3, type=int)
    parser.add_argument("--light_steps", default=6, type=int)
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()

    directory = Path(args.directory)
    with open(directory / "materials.pkl", "rb") as filedata:
        materials = pickle.load(filedata)
    figures_directory = Path(paz.directory.make(directory / "figures"))
    figures.configure()
    quality = (args.image_size, args.supersampling, args.light_steps)
    key = jax.random.key(args.seed)
    write_material_figures(key, materials, figures_directory, *quality)
