import matplotlib.pyplot as plt
import jax.numpy as jp
import paz
from paz import SE3
from paz.graphics.types import PointLight, Material, Shape
from paz.graphics.constants import CUBE, PLANE, SPHERE
from paz.graphics import camera
from paz.graphics.renderer import render

# Materials
mirror_material = Material(
    color=jp.array([1.0, 1.0, 1.0]),
    ambient=0.1,
    diffuse=0.1,  # Low diffuse for mirror
    specular=0.9,
    shininess=200.0,
    reflective=0.8,
)

glass_material = Material(
    color=jp.array([0.9, 0.9, 1.0]),
    ambient=0.1,
    diffuse=0.1,
    specular=0.9,
    shininess=200.0,
    refractive=0.9,
    refractive_index=1.5,
)

diffuse_material = Material(
    color=jp.array([0.8, 0.2, 0.2]),  # Red
    ambient=0.1,
    diffuse=0.9,
    specular=0.1,
    shininess=50.0,
)

floor_material = Material(
    color=jp.array([0.5, 0.5, 0.5]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.1,
    shininess=50.0,
)

# Objects
floor = Shape(jp.eye(4), PLANE, floor_material)

sphere_pose = SE3.translation(jp.array([-1.5, 1.0, 0.0]))
sphere = Shape(sphere_pose, SPHERE, mirror_material)

cube_pose = SE3.translation(jp.array([1.5, 1.0, 0.0]))
cube = Shape(cube_pose, CUBE, glass_material)

wall_pose = SE3.translation(jp.array([0.0, 2.0, -4.0])) @ SE3.rotation_x(
    jp.pi / 2
)
# wall = Shape(wall_pose, PLANE, diffuse_material)

# Camera
camera_pose = SE3.view_transform(
    camera_origin=jp.array([0.0, 3.0, 6.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

lights = [
    PointLight(
        jp.array([1.0, 1.0, 1.0]),
        jp.array([0.0, 10.0, 5.0]),
    )
]

H, W = 120, 160
rays = camera.build_rays((H, W), jp.pi / 3.0, camera_pose)

# Render
image, depth = render(
    (H, W),
    camera_pose,
    rays,
    paz.graphics.Scene([floor, sphere, cube]),
    lights,
    mask=None,
    shadows=True,  # Shadows off for speed/simplicity first
)

image = paz.image.denormalize(image)
plt.imshow(image)
plt.savefig("reflection_test.png")
print("Rendered reflection_test.png")
paz.graphics.viewer
