import jax
import jax.numpy as jp
import paz
from paz.graphics.mesh import (
    Mesh,
    build_sphere,
    merge_meshes,
    render,
    tile_render,
)
from paz.graphics.types import PointLight, Material
from paz.graphics.camera import build_rays

H, W = 2**8, 2**8
y_FOV = jp.pi / 3.0

camera_origin = jp.array([0.0, 2.0, -4.0])
camera_pose = paz.SE3.view_transform(
    camera_origin, jp.zeros(3), jp.array([0.0, 1.0, 0.0])
)

lights = [
    PointLight(jp.ones(3) * 0.8, jp.array([5.0, 8.0, 5.0])),
    PointLight(jp.ones(3) * 0.4, jp.array([-5.0, 4.0, 3.0])),
]

vertices, faces, edges = build_sphere(radius=1.0, subdivisions=5)
print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

red = jp.array([0.8, 0.2, 0.1])
vertex_colors = jp.repeat(red[None], len(vertices), axis=0)
material = Material(jp.zeros(3), 0.1, 0.7, 0.5, 100.0)
transform = paz.SE3.translation(jp.zeros(3))
mesh = Mesh(vertices, vertex_colors, transform, material, faces, edges)
meshes, mask = merge_meshes(mesh)

rays = build_rays((H, W), y_FOV, camera_pose)
render_fn = jax.jit(render, static_argnums=(0,))
image, depth = render_fn((H, W), camera_pose, rays, meshes, mask, lights)
image = paz.image.denormalize(image)
paz.image.write("render_full.png", image)
print("Saved render_full.png")

tile_render_fn = jax.jit(tile_render, static_argnums=(0, 2, 3))
tile_shape = (4, 4)
image, depth = tile_render_fn(
    tile_shape, y_FOV, H, W, camera_pose, meshes, mask, lights
)
image = paz.image.denormalize(image)
paz.image.write("render_tiled.png", image)
print("Saved render_tiled.png")

print("Computing gradient through vertex positions...")


def loss_fn(verts):
    mesh = Mesh(verts, vertex_colors, transform, material, faces, edges)
    meshes, mask = merge_meshes(mesh)
    _, depth = render_fn((H, W), camera_pose, rays, meshes, mask, lights)
    return jp.sum(depth)


grad_fn = jax.jit(jax.grad(loss_fn))
grads = grad_fn(vertices)
print(f"Gradient shape: {grads.shape}")
print(f"Gradient norm: {jp.linalg.norm(grads):.4f}")
print("Done.")
