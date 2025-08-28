# TODO put another cylinder next to it to make sure is of length 1 total
# TODO should group has its own pose?
# TODO make logic to check that parent array has the same length as shapes
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

line_width = 0.05
line_scale = paz.SE3.scaling(jp.array([line_width, 1.0, line_width]))
line_shift = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))
line_transform = line_shift @ line_scale


head_scale = paz.SE3.scaling(
    jp.array([2 * line_width, 2 * line_width, 2 * line_width])
)
head_shift = paz.SE3.translation(jp.array([0.0, 0.0, 0.0]))
head_transform = head_shift @ head_scale

origin_scale = paz.SE3.scaling(jp.full((3,), line_width * 2.0))
origin = paz.graphics.Sphere(origin_scale, O_material)

line = paz.graphics.Cylinder(line_transform, G_material)
head = paz.graphics.Cone(head_transform, G_material)
arrow = paz.graphics.Group([line, head], jp.array([-1, -1]))

arrow = [
    # paz.graphics.Cylinder(line_transform, G_material),
    arrow,
    origin,
    # paz.graphics.Plane(),
]

camera_pose = paz.SE3.view_transform(
    camera_origin=jp.array([0.0, 2.0, 5.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

H, W = 480, 640
y_FOV = jp.pi / 4.0
light = paz.graphics.PointLight(jp.ones(3), jp.array([-4.0, 5.0, 6.0]))
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
paz.partial(paz.graphics.render, (H, W))
render = jax.jit(paz.graphics.render, static_argnums=(0,))
image, depth = render((H, W), camera_pose, rays, arrow, light)
paz.image.show(paz.image.denormalize(image))
