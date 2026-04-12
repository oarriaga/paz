# import jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jp
import paz

blue_material = paz.graphics.Material(color=jp.array([0.36, 0.77, 0.96]))
red_material = paz.graphics.Material(color=paz.graphics.RED)
black_material = paz.graphics.Material(color=jp.full(3, 0.2))
white_material = paz.graphics.Material(color=paz.graphics.WHITE)
grey_material = paz.graphics.Material(color=jp.array([0.65, 0.65, 0.65]))

body_scale = paz.SE3.scaling(jp.array([2.5, 1.5, 0.8]))
body = paz.graphics.Cube(body_scale, blue_material)

lens_shift = paz.SE3.translation(jp.array([0.0, 0.0, 1.0]))
lens_scale = paz.SE3.scaling(jp.array([1.3, 1.3, 0.2]))
lens_angle = paz.SE3.rotation_x(jp.pi / 2)
lens_base = paz.graphics.Cylinder(
    lens_shift @ lens_scale @ lens_angle, grey_material
)

lens_barrel_transform = (
    paz.SE3.translation(jp.array([0.0, 0.0, 1.2]))
    @ paz.SE3.scaling(jp.array([1.1, 1.1, 0.1]))
    @ paz.SE3.rotation_x(jp.pi / 2)
)
lens_barrel = paz.graphics.Cylinder(lens_barrel_transform, black_material)

lens_element_transform = paz.SE3.translation(
    jp.array([0.0, 0.0, 1.3])
) @ paz.SE3.scaling(jp.array([0.9, 0.9, 0.05]))
lens_element = paz.graphics.Sphere(
    transform=lens_element_transform, material=white_material
)

button_shift = paz.SE3.translation(jp.array([2.0, 1.5, 0.0]))
button_scale = paz.SE3.scaling(jp.array([0.2, 0.2, 0.2]))
button = paz.graphics.Cylinder(button_shift @ button_scale, red_material)

viewfinder_shift = paz.SE3.translation(jp.array([1.9, 1.1, 1.0]))
viewfinder_scale = paz.SE3.scaling(jp.array([0.4, 0.2, 0.1]))
viewfinder = paz.graphics.Cube(
    viewfinder_shift @ viewfinder_scale, white_material
)

camera = paz.graphics.Group(
    [
        body,
        lens_base,
        lens_barrel,
        lens_element,
        button,
        viewfinder,
    ]
)

scene = paz.graphics.Scene(nodes=[camera])
world_to_camera = paz.SE3.view_transform(
    camera_origin=jp.array([0.0, 8.0, 8.0]),
    target_origin=jp.array([0.0, 0.5, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)
paz.graphics.scene.show(scene)
paz.graphics.viewer(scene, world_to_camera, True)
