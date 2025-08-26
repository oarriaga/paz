import jax.numpy as jp
import matplotlib.pyplot as plt
from paz import SE3

import paz
from paz.graphics.types import PointLight, Material, Shape, Pattern, Group
from paz.graphics.constants import SPHERE, CYLINDER, CONE

# Assuming the staging functions are now integrated or available
from paz.graphics import camera
from paz.graphics import render

H, W = 480, 640
y_FOV = jp.pi / 4.0

grey_material = Material(
    color=jp.array([0.75, 0.75, 0.75]),
    ambient=0.4,
    diffuse=0.5,
    specular=0.1,
    shininess=50.0,
)
red_material = Material(
    color=jp.array([1.0, 0.0, 0.0]),
    ambient=0.2,
    diffuse=0.9,
    specular=0.3,
    shininess=200.0,
)
green_material = Material(
    color=jp.array([0.0, 1.0, 0.0]),
    ambient=0.2,
    diffuse=0.9,
    specular=0.3,
    shininess=200.0,
)
blue_material = Material(
    color=jp.array([0.0, 0.0, 1.0]),
    ambient=0.2,
    diffuse=0.9,
    specular=0.3,
    shininess=200.0,
)
default_pattern = Pattern()

# The sphere is the root of our group
sphere = Shape(
    transform=SE3.scaling(jp.array([1.0, 1.0, 1.0])),
    type=SPHERE,
    material=grey_material,
)

cylinder_base_translation = SE3.translation(jp.array([0.0, 2.0, 0.0]))
cylinder_scaling = SE3.scaling(jp.array([0.25, 2.0, 0.25]))

cone_tip_translation = SE3.translation(jp.array([0.0, 5.0, 0.0]))
cone_scaling = SE3.scaling(jp.array([0.5, 1.0, 0.5]))


# Red X-Axis
cylinder_x = Shape(
    transform=SE3.rotation_z(-jp.pi / 2.0)
    @ cylinder_base_translation
    @ cylinder_scaling,
    type=CYLINDER,
    material=red_material,
)
cone_x = Shape(
    transform=SE3.rotation_z(-jp.pi / 2.0)
    @ cone_tip_translation
    @ cone_scaling,
    type=CONE,
    material=red_material,
)

# Green Y-Axis
cylinder_y = Shape(
    transform=cylinder_base_translation @ cylinder_scaling,
    type=CYLINDER,
    material=green_material,
)
cone_y = Shape(
    transform=cone_tip_translation @ cone_scaling,
    type=CONE,
    material=green_material,
)

# Blue Z-Axis
cylinder_z = Shape(
    transform=SE3.rotation_x(jp.pi / 2.0)
    @ cylinder_base_translation
    @ cylinder_scaling,
    type=CYLINDER,
    material=blue_material,
)
cone_z = Shape(
    transform=SE3.rotation_x(jp.pi / 2.0) @ cone_tip_translation @ cone_scaling,
    type=CONE,
    material=blue_material,
)

# The list of shapes now contains all components
shapes_list = [
    sphere,
    cylinder_x,
    cone_x,
    cylinder_y,
    cone_y,
    cylinder_z,
    cone_z,
]

parent_array = jp.array([-1, 0, 0, 0, 0, 0, 0])
axes = Group(shapes_list, parent_array)

paz.graphics.save("axes.json", group=axes)
# axes = paz.graphics.load("axes.json")

camera_pose = SE3.view_transform(
    camera_origin=jp.array([0.0, 10.0, 10.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

lights = [
    PointLight(
        intensity=jp.array([1.0, 1.0, 1.0]), position=jp.array([-4.0, 5.0, 6.0])
    )
]

rays = camera.build_rays((H, W), y_FOV, camera_pose)

print("Preparing scene graph and rendering...")

image_data, _ = render(
    image_shape=(H, W),
    world_to_camera=camera_pose,
    rays=rays,
    shapes=axes,
    lights=lights,
)

print("Rendering complete. Displaying image...")

image_data = jp.clip(image_data, 0, 1)
plt.imshow(image_data)
plt.axis("off")
plt.show()
