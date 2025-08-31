# should be an inherit property of the shapes?
# and transform by pose? an element of SE3
# we should be able to do groups of groups (hierarchical models)
from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax
import jax.numpy as jp
import paz

set_cache_dir(paz.logger.make_directory("cache"))

material = [0.2, 0.9, 0.3, 200.0]
R_material = paz.graphics.Material(paz.graphics.RED, *material)
G_material = paz.graphics.Material(paz.graphics.GREEN, *material)
B_material = paz.graphics.Material(paz.graphics.BLUE, *material)
O_material = paz.graphics.Material(jp.array([0.75, 0.75, 0.75]), *material)

origin = paz.graphics.Sphere(paz.SE3.scaling(jp.full((3,), 1.0)), O_material)

line_scale = paz.SE3.scaling(jp.array([0.25, 2.0, 0.25]))
line_shift = paz.SE3.translation(jp.array([0.0, 2.0, 0.0]))
line_transform = line_shift @ line_scale

head_shift = paz.SE3.translation(jp.array([0.0, 5.0, 0.0]))
head_scale = paz.SE3.scaling(jp.array([0.5, 1.0, 0.5]))
head_transform = head_shift @ head_scale

line_y = paz.graphics.Cylinder(line_transform, G_material)
head_y = paz.graphics.Cone(head_transform, G_material)

rotate_in_z = paz.SE3.rotation_z(-jp.pi / 2.0)
line_x = paz.graphics.Cylinder(rotate_in_z @ line_transform, R_material)
head_x = paz.graphics.Cone(rotate_in_z @ head_transform, R_material)

rotate_in_x = paz.SE3.rotation_x(jp.pi / 2.0)
line_z = paz.graphics.Cylinder(rotate_in_x @ line_transform, B_material)
head_z = paz.graphics.Cone(rotate_in_x @ head_transform, B_material)

sphere = paz.graphics.Sphere(paz.SE3.translation(jp.array([0.0, 2.0, 0.0])))

axes = paz.graphics.Group(
    [origin, line_x, head_x, line_y, head_y, line_z, head_z]
)

paz.graphics.save("axes.json", axes)

scene = paz.graphics.Scene([axes])

camera_pose = paz.SE3.view_transform(
    camera_origin=jp.array([0.0, 0.0, 10.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

H, W = 480, 640
y_FOV = jp.pi / 4.0
light = paz.graphics.PointLight(jp.ones(3), jp.array([-4.0, 5.0, 6.0]))
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
render = jax.jit(paz.graphics.render, static_argnums=(0,))
image, depth = render((H, W), camera_pose, rays, scene, light)
paz.image.show(paz.image.denormalize(image))
