import jax.numpy as jp
import jax
import paz
from paz import SE3
from paz.graphics import Sphere, SphericalPattern

# GLOBE

checkboard = True
GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


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
pattern_scale = SE3.scaling(jp.array([1.0, 1.0, 1.0]))
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)


if checkboard:
    pattern_scale = SE3.scaling(jp.full(3, 1.05))
    # pattern_transform = SE3.rotation_z(-jp.deg2rad(30)) @ pattern_scale
    globe_pattern = SphericalPattern(squares, pattern_scale)
    globe_material = zero_material
else:
    image = paz.image.normalize(paz.image.load("earthmap1k.jpg"))
    image = paz.image.flip_left_right(image)
    angle = SE3.rotation_y(jp.deg2rad(-10))
    globe_pattern = SphericalPattern(
        image, SE3.rotation_x(jp.deg2rad(30)) @ angle
    )
    globe_material = zero_material

globe = Sphere(SE3.scaling(jp.full(3, 1.05)), globe_material, globe_pattern)


scene = paz.graphics.Scene([globe])

camera_pose = SE3.view_transform(
    jp.array([0.0, 0.0, 3.0]),
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
image_name = "earth_checkboard.png" if checkboard else "earth.png"
paz.image.write(image_name, image)
