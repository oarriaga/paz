from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz
from paz import SE3
from paz.graphics import PointLight, Material, Cylinder, Cone
import matplotlib.pyplot as plt

set_cache_dir(paz.logger.make_directory("cache"))


blue_material = Material(jp.array([0.36, 0.77, 0.96]), 0.2, 0.9, 0.1, 50.0)
gold_material = Material(jp.array([0.9, 0.7, 0.2]), 0.3, 0.8, 0.8, 200.0)
base_material = Material(jp.array([0.6, 0.5, 0.5]), 0.1, 0.7, 0.1, 20.0)


def Base(transform, base_material, accent_material):
    base_shapes = []

    bottom_shift = SE3.translation([0.0, 0.0, 0.0])
    bottom_scale = SE3.scaling([1.0, 0.05, 1.0])
    bottom_disk = Cylinder(bottom_shift @ bottom_scale, base_material)
    base_shapes.append(bottom_disk)

    concave_shift = SE3.translation([0.0, 0.25, 0.0])
    concave_scale = SE3.scaling([0.975, 0.2, 0.975])
    concave_curve = Cone(concave_shift @ concave_scale, base_material)
    base_shapes.append(concave_curve)

    middle_shift = SE3.translation([0.0, 0.15, 0.0])
    middle_scale = SE3.scaling([0.7, 0.05, 0.7])
    middle_disk = Cylinder(middle_shift @ middle_scale, base_material)
    base_shapes.append(middle_disk)

    upper_shift = SE3.translation([0.0, 0.40, 0.0])
    upper_scale = SE3.scaling([0.675, 0.2, 0.675])
    upper_curve = Cone(upper_shift @ upper_scale, base_material)
    base_shapes.append(upper_curve)

    neck_shift = SE3.translation([0.0, 0.425, 0.0])
    neck_scale = SE3.scaling([0.2, 0.20, 0.2])
    neck_curve = Cylinder(neck_shift @ neck_scale, base_material)
    base_shapes.append(neck_curve)

    return paz.graphics.Group(shapes=base_shapes, transform=transform)


def Segment(segment_arg, scaling, num_segments, arc_radius, material):
    angle = (segment_arg / num_segments) * jp.pi - (jp.pi / 2.0)
    x = arc_radius * jp.cos(angle)
    z = arc_radius * jp.sin(angle)
    position = SE3.translation(jp.array([x, 0.0, z]))
    rotation = SE3.rotation_y(-angle + (jp.pi / 2.0)) @ SE3.rotation_z(
        jp.pi / 2.0
    )
    return paz.graphics.Cylinder(position @ rotation @ scaling, material)


def Arc(radius, num_segments, segment_length, segment_radius, material):
    segment_size = SE3.scaling([segment_radius, segment_length, segment_radius])
    _segment = paz.lock(Segment, segment_size, num_segments, radius, material)
    segments = [_segment(arg) for arg in range(num_segments + 1)]
    standup = SE3.rotation_x(jp.pi / 2.0)
    rotate = SE3.rotation_z(jp.deg2rad(-30))
    return paz.graphics.Group(segments, rotate @ standup)


arc = Arc(1.175, 20, 0.1, 0.06, gold_material)
globe = paz.graphics.Sphere(SE3.scaling(jp.full(3, 1.05)), blue_material)
base = Base(SE3.translation([0.0, -1.7, 0.0]), base_material, gold_material)


def Button(radius=1.1, angle=60, height=0.075, size=0.05):
    material = Material(jp.array([0.3, 0.3, 0.3]))
    x = jp.cos(jp.deg2rad(angle))
    y = jp.sin(jp.deg2rad(angle))
    shift = SE3.translation(radius * jp.array([x, y, 0]))
    scale = paz.SE3.scaling(jp.array([size, height, size]))
    angle = paz.SE3.rotation_z(-jp.deg2rad(30))
    # transform = button_shift @ button_angle @ button_scale, material
    bottom = paz.graphics.Cylinder(shift @ angle @ scale, material)
    top_shift = SE3.translation((radius + height) * jp.array([x, y, 0]))
    top = paz.graphics.Sphere(top_shift @ angle @ scale, material)
    return paz.graphics.Group([bottom, top], jp.eye(4))


top_button = Button(angle=60)
bottom_button = Button(angle=240)
scene = paz.graphics.Scene([base, globe, arc, top_button, bottom_button])

camera_pose = SE3.view_transform(
    jp.array([0.0, 4.0, 4.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = PointLight(jp.array([1.5, 1.5, 1.5]), jp.array([5.0, 4.0, 8.0]))

H, W = 720, 1024
y_FOV = jp.pi / 4.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        world_to_camera=camera_pose,
        rays=rays,
        lights=lights,
    )
)
# image, depth = render(scene=scene)
# plt.imshow(jp.clip(image, 0, 1))
# plt.axis("off")
# plt.show()
paz.graphics.viewer(scene, camera_pose)
