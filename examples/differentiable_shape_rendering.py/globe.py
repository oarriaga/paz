from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz
from paz import SE3
from paz.graphics import (
    PointLight,
    Material,
    Cylinder,
    Pattern,
    SphericalPattern,
    PlanarPattern,
    Cone,
    Cube,
    WHITE,
    BLACK,
    PLANAR_PATTERN,
)

set_cache_dir(paz.logger.make_directory("cache"))


def CheckeredImage(box_size=50, rows=8, cols=8, color_A=BLACK, color_B=WHITE):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


def Button(material, radius=1.1, angle=60, height=0.075, size=0.05):
    x = jp.cos(jp.deg2rad(angle))
    y = jp.sin(jp.deg2rad(angle))
    shift = SE3.translation(radius * jp.array([x, y, 0]))
    scale = paz.SE3.scaling(jp.array([size, height, size]))
    angle = paz.SE3.rotation_z(-jp.deg2rad(30))
    bottom = paz.graphics.Cylinder(shift @ angle @ scale, material)
    top_shift = SE3.translation((radius + height) * jp.array([x, y, 0]))
    top = paz.graphics.Sphere(top_shift @ angle @ scale, material)
    return paz.graphics.Group([bottom, top], jp.eye(4))


def Base(transform, base_material, pattern, pattern_neck):
    bottom_shift = SE3.translation([0.0, 0.0, 0.0])
    bottom_scale = SE3.scaling([1.0, 0.05, 1.0])
    bottom_disk = Cylinder(bottom_shift @ bottom_scale, base_material, pattern)

    concave_shift = SE3.translation([0.0, 0.25, 0.0])
    concave_scale = SE3.scaling([0.975, 0.2, 0.975])
    concave_curve = Cone(concave_shift @ concave_scale, base_material, pattern)

    middle_shift = SE3.translation([0.0, 0.15, 0.0])
    middle_scale = SE3.scaling([0.7, 0.05, 0.7])
    middle_disk = Cylinder(middle_shift @ middle_scale, base_material, pattern)

    upper_shift = SE3.translation([0.0, 0.40, 0.0])
    upper_scale = SE3.scaling([0.675, 0.2, 0.675])
    upper_curve = Cone(upper_shift @ upper_scale, base_material, pattern)

    neck_shift = SE3.translation([0.0, 0.425, 0.0])
    neck_scale = SE3.scaling([0.2, 0.20, 0.2])

    neck_curve = Cylinder(neck_shift @ neck_scale, base_material, pattern_neck)
    shapes = [bottom_disk, concave_curve, middle_disk, upper_curve, neck_curve]
    return paz.graphics.Group(shapes, transform)


def Segment(segment_arg, scaling, num_segments, arc_radius, material, pattern):
    angle = (segment_arg / num_segments) * jp.pi - (jp.pi / 2.0)
    x = arc_radius * jp.cos(angle)
    z = arc_radius * jp.sin(angle)
    position = SE3.translation(jp.array([x, 0.0, z]))
    rotation_y = SE3.rotation_y(-angle + (jp.pi / 2.0))
    rotation_z = SE3.rotation_z(jp.pi / 2.0)
    rotation = rotation_y @ rotation_z
    return Cylinder(position @ rotation @ scaling, material, pattern)


def Arc(
    radius, num_segments, segment_length, segment_radius, material, pattern
):
    pattern = pattern._replace(type=paz.graphics.PLANAR_PATTERN)
    segment_size = SE3.scaling([segment_radius, segment_length, segment_radius])
    _segment = paz.lock(
        Segment, segment_size, num_segments, radius, material, pattern
    )
    segments = [_segment(arg) for arg in range(num_segments + 1)]
    standup = SE3.rotation_x(jp.pi / 2.0)
    rotate = SE3.rotation_z(jp.deg2rad(-30))
    return paz.graphics.Group(segments, rotate @ standup)


checkboard = True
GREEN = (0.415, 0.749, 0.639)
GRAY = (0.75, 0.75, 0.75)

blue_material = Material(jp.array([0.36, 0.77, 0.96]), 0.2, 0.9, 0.1, 50.0)
gold_material = Material(jp.array([0.9, 0.7, 0.2]), 0.3, 0.8, 0.8, 200.0)
base_material = Material(jp.array([0.6, 0.5, 0.5]), 0.1, 0.7, 0.1, 20.0)
zero_material = Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)
ones_material = Material(jp.ones(3), 0.85, 0.1, 0.0, 100)


squares = CheckeredImage(25, 20, 20, GREEN, WHITE)
pattern_scale = SE3.scaling(jp.full(3, 1.05))
pattern_transform = SE3.rotation_z(-jp.deg2rad(30)) @ pattern_scale
checkboard_pattern = SphericalPattern(squares, pattern_transform)


pattern_scale = SE3.scaling(jp.full(3, 40.0))
pattern_transform = SE3.rotation_z(-jp.deg2rad(30)) @ pattern_scale
checkboard_pattern_planar = SphericalPattern(squares, pattern_transform)

checkboard_pattern_2 = PlanarPattern(squares, SE3.scaling(jp.full(3, 4.1)))
zero_pattern = Pattern()

# ARC
if checkboard:
    arc = Arc(1.175, 20, 0.1, 0.06, zero_material, checkboard_pattern_planar)
else:
    arc = Arc(1.175, 20, 0.1, 0.06, gold_material, zero_pattern)

# GLOBE
globe_scale = SE3.scaling(jp.full(3, 1.05))

if checkboard:
    globe = paz.graphics.Sphere(globe_scale, zero_material, checkboard_pattern)
else:
    globe = paz.graphics.Sphere(globe_scale, blue_material)


# BASE
squares = CheckeredImage(100, 10, 10, GREEN, WHITE)
pattern_scale = SE3.scaling(jp.array([1.0, 1.0, 1.0]))
pattern_neck = SphericalPattern(squares, pattern_scale)
if checkboard:
    base = Base(
        SE3.translation([0.0, -1.7, 0.0]),
        zero_material,
        checkboard_pattern_2,
        pattern_neck,
    )
else:
    base = Base(
        SE3.translation([0.0, -1.7, 0.0]),
        gold_material,
        zero_pattern,
        zero_pattern,
    )

# BUTTONS
if checkboard:
    button_material = ones_material
else:
    button_material = Material(jp.array([0.3, 0.3, 0.3]))

top_button = Button(button_material, angle=60)
bottom_button = Button(button_material, angle=240)

# FLOOR
floor_image = CheckeredImage(50, 8, 8, WHITE, GRAY)
floor_pattern = paz.graphics.Pattern(jp.eye(4), PLANAR_PATTERN, floor_image)
floor_shift = SE3.translation(jp.array([0.0, -1.725, 0.0]))
floor_scale = SE3.scaling(jp.array([10.0, 0.05, 10.0]))
floor = Cube(floor_shift @ floor_scale, zero_material, floor_pattern)

# WALL
wall_image = CheckeredImage(50, 8, 8, WHITE, GRAY)
wall_pattern = PlanarPattern(wall_image, SE3.scaling(jp.array([1.0, 1.0, 1.0])))
wall_scale = SE3.scaling(jp.array([10.0, 0.05, 10.0]))
wall_angle = SE3.rotation_x(jp.pi / 2.0)
wall_shift = SE3.translation(jp.array([1.25, 2.1, -5.0]))
wall_transform = wall_shift @ wall_angle @ wall_scale
wall = paz.graphics.Cube(wall_transform, zero_material, wall_pattern)

shapes = [base, globe, arc, top_button, bottom_button, floor, wall]
scene = paz.graphics.Scene(shapes)

camera_pose = SE3.view_transform(
    jp.array([0.0, 4.0, 4.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = PointLight(jp.array([1.5, 1.5, 1.5]), jp.array([5.0, 4.0, 8.0]))

H, W = 720 // 4, 1024 // 4
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
paz.graphics.viewer(scene, camera_pose)
