from functools import partial
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.image import imread

from paz import SE3
from paz.graphics import PointLight, Material, Pattern, Shape
from paz.graphics.camera import build_rays
from paz.graphics import render_with_shadows
from paz.graphics import (
    SPHERE,
    CUBE,
    CONE,
    CYLINDER,
    PLANE,
    NO_PATTERN,
    PLANAR_PATTERN,
    SPHERICAL_PATTERN,
)

GREEN = jp.array([0.0, 1.0, 0.0])
BLUE = jp.array([0.0, 0.0, 1.0])
PINK = jp.array([2.0, 0.0, 1.0])


def merge_leaf(*leafs):
    concatenated_leafs = []
    for leaf in leafs:
        concatenated_leafs.append(leaf)
    return jp.array(concatenated_leafs)


def merge(*pytrees):
    return jax.tree.map(merge_leaf, *pytrees)


lights = [
    PointLight(jp.array([0.4, 0.4, 0.4]), jp.array([-10.0, 10.0, -10.0])),
    PointLight(jp.array([0.4, 0.4, 0.4]), jp.array([+10.0, 10.0, -10.0])),
]


camera_pose = SE3.view_transform(
    jp.array([0, 4.0, -3.0]),
    jp.array([0.0, 0.0, 4.0]),
    jp.array([0.0, 1.0, 0.0]),
)
H, W = image_size = 480, 640
rays = build_rays(image_size, jp.pi / 3, camera_pose)

rotate = SE3.rotation_y(-jp.pi / 3)

# left-right, up-down, backward-forward
translate_01 = SE3.translation(jp.array([3.0, 1.5, 5.0]))
translate_02 = SE3.translation(jp.array([-3.0, 1.5, 5.0]))
translate_03 = SE3.translation(jp.array([0.0, 1.5, 5.0]))
translate_04 = SE3.translation(jp.array([-2.0, 1.5, 2.0]))

scale_01 = SE3.scaling(jp.array([1.5, 1.5, 1.5]))
scale_02 = SE3.scaling(jp.array([1.0, 1.0, 1.0]))
scale_03 = SE3.scaling(jp.array([0.5, 0.5, 0.5]))
scale_04 = SE3.scaling(jp.array([0.75, 2.0, 0.75]))

transform_01 = translate_01 @ rotate @ scale_01
transform_02 = translate_02 @ rotate @ scale_02
transform_03 = translate_03 @ rotate @ scale_03
transform_04 = translate_04 @ rotate @ scale_04
transform_05 = jp.eye(4)

material_01 = Material(jp.full(3, 0.0), 0.0, 0.9, 0.0, 10)
material_02 = Material(GREEN, 0.1, 0.9, 0.9, 200.0)
material_03 = Material(BLUE, 0.1, 0.9, 0.9, 200.0)
material_04 = Material(PINK, 0.1, 0.9, 0.9, 200.0)
material_05 = Material(jp.array([1.0, 0.9, 0.9]), 0.1, 0.9, 0.0, 200.0)

empty_image = jp.zeros((500, 1000, 3))
pattern_01 = Pattern(jp.eye(4), NO_PATTERN, empty_image)

earth_image = jp.array(imread("earthmap1k.jpg")) / 255.0
pattern_02 = Pattern(jp.eye(4), SPHERICAL_PATTERN, earth_image)

floor_image = jp.array(imread("wood_uv.png"))
floor_image = floor_image[:500, :1000, :]
pattern_03 = Pattern(SE3.scaling(jp.full(3, 10.0)), PLANAR_PATTERN, floor_image)

shape_01 = Shape(transform_01, SPHERE, pattern_02, material_01)
shape_02 = Shape(transform_02, CUBE, pattern_01, material_02)
shape_03 = Shape(transform_03, CYLINDER, pattern_01, material_03)
shape_04 = Shape(transform_04, CONE, pattern_01, material_04)
shape_05 = Shape(transform_05, PLANE, pattern_03, material_01)

scene = merge(shape_01, shape_02, shape_03, shape_04, shape_05)
mask = jp.ones(shape=(5,), dtype=bool)
shadows = True

render = partial(render_with_shadows, (H, W), camera_pose, rays)
fast_render = jax.jit(render)
image, depth = fast_render(scene, mask, lights)
plt.imshow(image)
plt.show()

mask = jp.array([False, True, False, True, True])
image, depth = fast_render(scene, mask, lights)
plt.imshow(image)
plt.show()

mask = jp.array([True, True, True, True, True])
image, depth = fast_render(scene, mask, lights)
plt.imshow(image)
plt.show()
