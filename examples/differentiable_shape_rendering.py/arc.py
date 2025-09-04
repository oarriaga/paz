import jax.numpy as jp
import jax
import paz
from paz import SE3
from paz.graphics import (
    Cylinder,
    Material,
    PlanarPattern,
    Pattern,
    SphericalPattern,
)

GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
explode = True


def CheckeredImage(box_size=50, rows=8, cols=8, color_A=GREEN, color_B=WHITE):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


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


checkboard = False
zero_material = Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)
zero_pattern = Pattern()
pattern_scale = SE3.scaling(jp.full(3, 40.0))
pattern_transform = SE3.rotation_z(-jp.deg2rad(180)) @ pattern_scale
squares = CheckeredImage(100, 10, 10, GREEN, WHITE)
checkboard_pattern_planar = SphericalPattern(squares, pattern_transform)
num_segments = 25 if explode else 50
segment_length = 0.06 if explode else 0.1


if checkboard:
    arc = Arc(
        1.175,
        num_segments,
        segment_length,
        0.06,
        zero_material,
        checkboard_pattern_planar,
    )
else:
    gold_color = jp.array([255, 215, 0]) / 255.0
    gold_material = Material(gold_color, 0.2, 0.1, 0.0, 100)
    gold_pattern = PlanarPattern(
        paz.image.normalize(paz.image.load("gold_uv.png")),
        SE3.scaling(jp.full(3, 50.0)),
    )
    zero = Material(jp.zeros(3), 0.7, 0.1, 0.0, 100)
    arc = Arc(1.175, num_segments, segment_length, 0.06, zero, gold_pattern)


scene = paz.graphics.Scene([arc])

camera_pose = SE3.view_transform(
    jp.array([0.0, 0.4, 3.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = paz.graphics.PointLight(jp.ones(3), jp.array([5.0, 4.0, 8.0]))

H, W = 1024, 1024
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

image, depth = render(scene=scene)
image = paz.image.resize_opencv(paz.image.denormalize(image), (H // 2, W // 2))
image_name = "arc_checkboard.png" if checkboard else "arc.png"
image_name = "exploded_" + image_name if explode else image_name
paz.image.write(image_name, image)
