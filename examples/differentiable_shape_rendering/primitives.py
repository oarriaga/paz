from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz


set_cache_dir(paz.logger.make_directory("cache"))

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
    jp.array([0.0, 8.0, 8.0]),
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
        shadows=True,
    )
)

checkered_image = CheckeredImage()
spherical_pattern = paz.graphics.SphericalPattern(checkered_image)
planar_pattern = paz.graphics.PlanarPattern(checkered_image)
cylindrical_pattern = paz.graphics.CylindricalPattern(
    checkered_image, paz.SE3.scaling(jp.full(3, 3.0))
)
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)

shape_01 = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([0.0, 1.0, -3.0])),
    zero_material,
    spherical_pattern,
)
shape_02 = paz.graphics.Cylinder(
    paz.SE3.translation(jp.array([0.0, 1.0, -1.0]))
    @ paz.SE3.scaling(jp.full(3, 1.0)),
    zero_material,
    cylindrical_pattern,
)
shape_03 = paz.graphics.Cone(
    paz.SE3.translation(jp.array([0.0, 1.0, 1.0])),
    zero_material,
    planar_pattern._replace(transform=paz.SE3.scaling(jp.full(3, 3.0))),
)
shape_04 = paz.graphics.Cube(
    paz.SE3.translation(jp.array([0.0, 1.0, 3.0]))
    @ paz.SE3.scaling(jp.full(3, 1.0)),
    zero_material,
    cylindrical_pattern._replace(transform=paz.SE3.scaling(jp.full(3, 4.0))),
)
floor = paz.graphics.Plane()

scene = paz.graphics.Scene([shape_01, shape_02, shape_03, shape_04, floor])
image, depth = render(scene=scene, mask=None)
paz.image.show(paz.image.denormalize(image))

key = jax.random.PRNGKey(0)
for key in jax.random.split(key, 100):
    mask = jax.random.randint(key, (4,), 0, 2)
    mask = jp.append(mask, 1)
    image, depth = render(scene=scene, mask=mask)
    paz.image.show(paz.image.denormalize(image))

# _scene, _lights, _mask = paz.graphics.scene.compile(scene, lights, None)

# image, depth = paz.graphics.render(
#     (H, W), camera_pose, rays, scene, lights, mask=None, shadows=True
# )
# paz.image.show(paz.image.denormalize(image))
# paz.graphics.viewer(scene, camera_pose)
