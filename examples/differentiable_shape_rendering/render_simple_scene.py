import matplotlib.pyplot as plt
import jax.numpy as jp
import paz
from paz import SE3
from paz.graphics.types import PointLight, Material, Shape
from paz.graphics.constants import CUBE, PLANE
from paz.graphics import camera
from paz.graphics import render

# from paz.graphics import render_with_shadows as render


red_material = Material(
    color=jp.array([1.0, 0.0, 0.0]),  # Red
    ambient=0.1,
    diffuse=0.9,
    specular=0.3,
    shininess=200.0,
)
grey_material = Material(
    color=jp.array([0.5, 0.5, 0.5]),  # Grey
    ambient=0.1,
    diffuse=0.9,
    specular=0.1,
    shininess=50.0,
)

floor_pose = SE3.rotation_x(jp.pi / 2.0)
floor_pose = jp.eye(4)

floor = Shape(floor_pose, PLANE, grey_material)
# cube_rotation = SE3.rotation_z(jp.pi / 4.0)
cube_pose = SE3.translation(jp.array([0.0, 1.0, 0.0]))
cube = Shape(cube_pose, CUBE, red_material)


camera_pose = SE3.view_transform(
    camera_origin=jp.array([3.0, 3.0, 0.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

lights = [
    PointLight(
        jp.array([1.0, 1.0, 1.0]),
        jp.array([-5.0, 5.0, -5.0]),
    )
]

H, W = 240, 320
rays = camera.build_rays((H, W), jp.pi / 3.0, camera_pose)
# import jax
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    image, depth = render(
        (H, W),
        camera_pose,
        rays,
        paz.graphics.Scene([floor, cube]),
        lights,
        None,
        False,
    )

image = paz.image.denormalize(image)
figure, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(depth, cmap="viridis")
plt.show()
