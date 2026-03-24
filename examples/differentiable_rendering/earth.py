import jax.numpy as jp
import paz

checkboard = False
GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
WHITE = (1.0, 1.0, 1.0)


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


squares = CheckeredImage(25, 20, 20, GREEN, WHITE)
pattern_scale = paz.SE3.scaling(jp.array([1.0, 1.0, 1.0]))
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)


if checkboard:
    pattern_scale = paz.SE3.scaling(jp.full(3, 1.05))
    globe_pattern = paz.graphics.SphericalPattern(squares, pattern_scale)
    globe_material = zero_material
else:
    image = paz.image.normalize(paz.image.load("earthmap1k.jpg"))
    image = paz.image.flip_left_right(image)
    angle = paz.SE3.rotation_y(jp.deg2rad(-10))
    globe_pattern = (image, paz.SE3.rotation_x(jp.deg2rad(30)) @ angle)
    globe_pattern = paz.graphics.SphericalPattern(*globe_pattern)
    globe_material = zero_material

globe = (paz.SE3.scaling(jp.full(3, 1.05)), globe_material, globe_pattern)

scene = paz.graphics.Scene([paz.graphics.Sphere(*globe)])

camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 0.0, 3.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

paz.graphics.viewer(scene, camera_pose)
