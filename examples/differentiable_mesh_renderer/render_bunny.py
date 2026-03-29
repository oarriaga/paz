from pathlib import Path
import os

# Keep the example runnable on machines where JAX GPU memory is already tight.
# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import paz
from paz.graphics.mesh import (
    Mesh,
    load_mesh,
    merge_meshes,
    tile_render,
)
from paz.graphics.types import Material, PointLight

H, W = 256, 256
TILE_SHAPE = (4, 4)
Y_FOV = jp.pi / 4.0


def normalize_vertices(vertices):
    bounds_min = jp.min(vertices, axis=0)
    bounds_max = jp.max(vertices, axis=0)
    center = (bounds_min + bounds_max) / 2.0
    max_extent = jp.max(bounds_max - bounds_min)
    return (vertices - center) / max_extent


def build_face_edges(faces):
    edge_pairs = [
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ]
    return jp.concatenate(edge_pairs, axis=0)


def make_bunny_mesh(path):
    vertices, faces, vertex_colors = load_mesh(path)
    vertices = normalize_vertices(vertices)
    edges = build_face_edges(faces)
    material = Material(jp.zeros(3), 0.15, 0.75, 0.25, 64.0)
    transform = paz.SE3.translation(
        jp.array([0.0, -0.12, 0.0])
    ) @ paz.SE3.rotation_y(jp.pi)
    return Mesh(vertices, vertex_colors, transform, material, faces, edges)


def main():
    example_dir = Path(__file__).resolve().parent
    mesh_path = example_dir / "bunny.obj"
    output_path = example_dir / "bunny_render.png"

    camera_origin = jp.array([1.3, 0.55, -2.2])
    camera_target = jp.array([0.0, -0.05, 0.0])
    camera_pose = paz.SE3.view_transform(
        camera_origin, camera_target, jp.array([0.0, 1.0, 0.0])
    )
    lights = [
        PointLight(jp.ones(3) * 1.4, camera_origin),
        PointLight(jp.array([0.6, 0.6, 0.7]), jp.array([-2.0, 2.0, -1.0])),
    ]

    bunny = make_bunny_mesh(mesh_path)
    meshes, mask = merge_meshes(bunny)

    render_fn = jax.jit(tile_render, static_argnums=(0, 2, 3))
    image, _ = render_fn(
        TILE_SHAPE, Y_FOV, H, W, camera_pose, meshes, mask, lights
    )
    image = paz.image.denormalize(image)
    paz.image.write(output_path, image)

    print(f"Loaded {mesh_path.name}")
    print(f"Saved render to {output_path}")


if __name__ == "__main__":
    main()
