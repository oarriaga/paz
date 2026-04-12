import matplotlib.pyplot as plt
import jax.numpy as jp
import jax
import paz


def build_centers(shape, sizes, x_center=-1.0, y_center=1.0):
    x_centers = jp.linspace(0, shape[0], shape[0], endpoint=False) * sizes[0]
    y_centers = jp.linspace(0, shape[1], shape[0], endpoint=False) * sizes[1]
    z_centers = jp.linspace(0, shape[2], shape[0], endpoint=False) * sizes[2]
    x_grid, y_grid, z_grid = jp.meshgrid(x_centers, y_centers, z_centers)
    x_grid = paz.algebra.to_column(x_grid) + x_center
    y_grid = paz.algebra.to_column(y_grid) + sizes[1]
    z_grid = paz.algebra.to_column(z_grid) + y_center
    centers = jp.concatenate([x_grid, y_grid, z_grid], axis=1)
    return centers


def build_transform(centers, scaling):
    transforms = []
    for center in centers:
        translate = paz.SE3.translation(center)
        scale = paz.SE3.scaling(scaling)
        transforms.append(translate @ scale)
    return jp.array(transforms)


def build_shapes(key, num_shapes, transforms):
    keys = jax.random.split(key, num_shapes)
    shapes = []
    for shape_arg in range(num_shapes):
        color = jax.random.uniform(keys[shape_arg], (3,), jp.float32, 0, 1)
        material = paz.graphics.Material(color=color)
        shapes.append(paz.graphics.Cube(transforms[shape_arg], material))
    return shapes


key = jax.random.PRNGKey(777)
shape = jp.full((3,), 7)
sizes = jp.full((3,), 0.25)
x_center = -1.0
y_center = 1.0
centers = build_centers(shape, sizes, x_center, y_center)
scaling = jp.array(sizes) / 2.0
transforms = build_transform(centers, scaling)
lights = [paz.graphics.PointLight(jp.ones(3), jp.array([-10.0, 10.0, -10.0]))]
world_to_camera = paz.SE3.view_transform(
    jp.array([0, 4.0, -5.0]),
    jp.array([0.0, 0.0, 4.0]),
    jp.array([0.0, 1.0, 0.0]),
)
H, W = image_size = 224, 224
rays = paz.graphics.camera.build_rays(image_size, jp.pi / 3, world_to_camera)
num_shapes = jp.prod(shape)
mask = jp.ones((num_shapes,), dtype=bool)
shapes = build_shapes(key, num_shapes, transforms)
scene = paz.graphics.Scene(shapes)
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        world_to_camera=world_to_camera,
        rays=rays,
        lights=lights,
        shadows=False,
    )
)

for subkey in jax.random.split(key, 20):
    mask = jax.random.randint(subkey, (num_shapes,), 0, 2)
    image, depth = render(scene=scene, mask=mask)
    plt.imshow(image)
    plt.title(f"Total Cubes: {jp.sum(mask)}")
    plt.show()
