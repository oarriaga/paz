import jax.numpy as jp
import paz

color_1_args = (jp.array([1.0, 0.0, 0.0]), 0.1, 0.9, 0.3, 200.0)
color_2_args = (jp.full(3, 0.75), 0.1, 0.9, 0.0, 200.0)
color_1 = paz.graphics.Material(*color_1_args)
color_2 = paz.graphics.Material(*color_2_args)
sphere_pose = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))
sphere = paz.graphics.Sphere(sphere_pose, color_1)
plane = paz.graphics.Plane(material=color_2)
scene = paz.graphics.Scene([sphere, plane])
camera_pose = paz.SE3.view_transform(
    camera_origin=jp.array([3.0, 3.0, 0.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)
lights = [paz.graphics.PointLight(jp.ones(3), jp.array([0.0, 3.0, 3.0]))]
paz.graphics.viewer(scene, camera_pose, True, lights, 240, 320, jp.pi / 3.0)
