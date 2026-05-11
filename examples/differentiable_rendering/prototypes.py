import jax.numpy as jp
import jax
import paz


GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
GRAY = (0.662, 0.647, 0.576)
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


camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 1.5, 2.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = paz.graphics.PointLight(jp.ones(3), jp.array([5.0, 4.0, 8.0]))

H, W = 1024, 1024
y_FOV = jp.pi / 4.0
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        shape=(H, W),
        y_FOV=y_FOV,
        pose=camera_pose,
        lights=lights,
        mask=None,
        tiles=(1, 1),
        chunk_size=1024,
    )
)

checkered_image = CheckeredImage()
spherical_pattern = paz.graphics.SphericalPattern(checkered_image)
planar_pattern = paz.graphics.PlanarPattern(checkered_image)
cylindrical_pattern = paz.graphics.CylindricalPattern(
    checkered_image, paz.SE3.scaling(jp.full(3, 3.0))
)
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)

shape_01 = paz.graphics.Sphere(jp.eye(4), zero_material, spherical_pattern)
shape_02 = paz.graphics.Cylinder(
    paz.SE3.scaling(jp.full(3, 0.7)), zero_material, cylindrical_pattern
)
shape_03 = paz.graphics.Cone(
    paz.SE3.translation(jp.array([0.0, 0.7, 0.0])),
    zero_material,
    planar_pattern._replace(transform=paz.SE3.scaling(jp.full(3, 3.0))),
)
shape_04 = paz.graphics.Cube(
    paz.SE3.scaling(jp.full(3, 0.7)),
    zero_material,
    cylindrical_pattern._replace(transform=paz.SE3.scaling(jp.full(3, 4.0))),
)

images = []
for shape_arg, shape in enumerate([shape_01, shape_02, shape_03, shape_04]):
    image, depth = render(scene=paz.graphics.Scene([shape]))
    image = paz.image.resize_opencv(
        paz.image.denormalize(jp.clip(image, 0.0, 1.0)), (H // 2, W // 2)
    )
    paz.image.write(f"prototype_{shape_arg}.png", image)
    images.append(image)

images = jp.array(images)
mosaic = paz.draw.mosaic(images, (1, len(images)), border=10)
paz.image.write("prototypes.png", mosaic)
