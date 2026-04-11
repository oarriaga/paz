# import jax

# jax.config.update("jax_platform_name", "cpu")
import matplotlib.pyplot as plt
import jax.numpy as jp
import paz
from paz.graphics import render

red = paz.graphics.Material(color=jp.array([1.0, 0.0, 0.0]), ambient=0.1, diffuse=0.9, specular=0.3, shininess=200.0)  # fmt: skip
grey = paz.graphics.Material(color=jp.full(3, 0.75), ambient=0.1, diffuse=0.9, specular=0.0, shininess=200.0)  # fmt: skip

sphere_pose = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))
shape = paz.graphics.Sphere(sphere_pose, red)

plane = paz.graphics.Plane(material=grey)
scene = paz.graphics.Scene([shape, plane])

camera_pose = paz.SE3.view_transform(
    camera_origin=jp.array([3.0, 3.0, 0.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)


H, W = 240, 320
lights = [paz.graphics.PointLight(jp.ones(3), jp.array([0.0, 3.0, 3.0]))]
paz.graphics.viewer(scene, camera_pose, True, lights, H, W, jp.pi / 3.0)
