import jax.numpy as jp
import jax
import paz
from paz import SE3
from paz.graphics import Cylinder, Cone, PlanarPattern, SphericalPattern

checkboard = False
explode = False
GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


def Base(transform, base_material, pattern, pattern_neck):
    if explode:
        distance = 0.33
        bottom_shift = SE3.translation([0.0, 0.0, 0.0])
        concave_shift = SE3.translation([0.0, 0.25 + distance, 0.0])
        middle_shift = SE3.translation([0.0, 0.15 + 2 * distance, 0.0])
        upper_shift = SE3.translation([0.0, 0.40 + 3 * distance, 0.0])
        neck_shift = SE3.translation([0.0, 0.425 + 4 * distance, 0.0])
    else:
        bottom_shift = SE3.translation([0.0, 0.0, 0.0])
        concave_shift = SE3.translation([0.0, 0.25, 0.0])
        middle_shift = SE3.translation([0.0, 0.15, 0.0])
        upper_shift = SE3.translation([0.0, 0.40, 0.0])
        neck_shift = SE3.translation([0.0, 0.425, 0.0])

    bottom_scale = SE3.scaling([1.0, 0.05, 1.0])
    bottom_disk = Cylinder(bottom_shift @ bottom_scale, base_material, pattern)

    concave_scale = SE3.scaling([0.975, 0.2, 0.975])
    concave_curve = Cone(concave_shift @ concave_scale, base_material, pattern)

    middle_scale = SE3.scaling([0.7, 0.05, 0.7])
    middle_disk = Cylinder(middle_shift @ middle_scale, base_material, pattern)

    upper_scale = SE3.scaling([0.675, 0.2, 0.675])
    upper_curve = Cone(upper_shift @ upper_scale, base_material, pattern)

    neck_scale = SE3.scaling([0.2, 0.20, 0.2])

    neck_curve = Cylinder(neck_shift @ neck_scale, base_material, pattern_neck)
    shapes = [bottom_disk, concave_curve, middle_disk, upper_curve, neck_curve]
    return paz.graphics.Group(shapes, transform)


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


pattern_scale = SE3.scaling(jp.array([1.0, 1.0, 1.0]))
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)

if checkboard:
    squares = CheckeredImage(100, 10, 10, GREEN, WHITE)
    neck_pattern = SphericalPattern(squares, pattern_scale)
    base_pattern = PlanarPattern(squares, SE3.scaling(jp.full(3, 4.1)))
    base_material = zero_material
else:
    base_material = paz.graphics.Material(jp.array(BLACK), 0.3, 0.7, 0.1, 20.0)
    image = paz.image.normalize(paz.image.load("wood_base.png"))[:800]
    shift = SE3.translation(jp.array([1.0, 0.0, 0.0]))
    base_pattern = PlanarPattern(image, shift)
    neck_pattern = PlanarPattern(image)
    # base_pattern = paz.graphics.Pattern()
    # neck_pattern = paz.graphics.Pattern()

base = Base(
    SE3.translation([0.0, -1.7, 0.0]),
    base_material,
    base_pattern,
    neck_pattern,
)


scene = paz.graphics.Scene([base])

camera_pose = SE3.view_transform(
    jp.array([0.0, -1.0, 3.1]),
    jp.array([0.0, -0.9, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)
if (not explode) and (not checkboard):
    camera_pose = SE3.translation([0.0, 0.5, 0.3]) @ camera_pose


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
image_name = "base_checkboard.png" if checkboard else "base.png"
image_name = "exploded_" + image_name if explode else image_name
paz.image.write(image_name, image)
