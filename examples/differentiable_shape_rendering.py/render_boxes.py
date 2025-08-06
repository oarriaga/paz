import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import paz
from paz import SE3


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
        translate = SE3.translation(center)
        scale = SE3.scaling(scaling)
        transforms.append(translate @ scale)
    return jp.array(transforms)


def build_shapes(key, num_shapes, transforms):
    shape_types = jp.full((num_shapes,), paz.graphics.CUBE)

    color = jax.random.uniform(key, (num_shapes, 3), jp.float32, 0, 1)
    ambient = jp.full((num_shapes,), 0.1)
    diffuse = jp.full((num_shapes,), 0.9)
    specular = jp.full((num_shapes,), 0.9)
    shininess = jp.full((num_shapes,), 200.0)
    material = paz.graphics.Material(
        color, ambient, diffuse, specular, shininess
    )

    empty_image = jp.zeros((num_shapes, 1, 1, 3))
    pattern_type = jp.full((num_shapes,), paz.graphics.NO_PATTERN)
    pattern_transform = jp.array([jp.eye(4) for shape_arg in range(num_shapes)])
    pattern = paz.graphics.Pattern(pattern_transform, pattern_type, empty_image)

    return paz.graphics.Shape(transforms, shape_types, pattern, material)


def get_neighbor_indices(shape):
    num_shapes = jp.prod(shape)
    indices = jp.arange(num_shapes).reshape(shape)

    def is_valid_neighbor(coord):
        i, j, k = coord
        return (
            (0 <= i)
            & (i < shape[0])
            & (0 <= j)
            & (j < shape[1])
            & (0 <= k)
            & (k < shape[2])
        )

    def get_valid_neighbors(x, y, z):
        neighbors = jp.array(
            [
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ],
            dtype=int,
        )
        valid_neighbors = jax.vmap(
            lambda coord: jax.lax.cond(
                is_valid_neighbor(coord),
                lambda x: x,
                lambda x: jp.array([-1, -1, -1]),
                coord,
            )
        )(neighbors)
        return valid_neighbors

    neighbor_indices = jax.vmap(
        jax.vmap(
            jax.vmap(get_valid_neighbors, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(jp.arange(shape[0]), jp.arange(shape[1]), jp.arange(shape[2]))
    return indices, neighbor_indices


def calculate_loss(indices, neighbor_indices, mask):
    def compute_loss_at(idx):
        x, y, z = jp.unravel_index(idx, indices.shape)
        neighbors = neighbor_indices[x, y, z]
        active_neighbors = jp.sum(
            jax.vmap(
                lambda n: jax.lax.cond(
                    n[0] != -1,
                    lambda _: mask[indices[n[0], n[1], n[2]]],
                    lambda _: 0,
                    operand=None,
                )
            )(neighbors)
        )
        return active_neighbors / 6.0

    loss = jax.vmap(compute_loss_at)(jp.arange(indices.size))
    return loss


key = jax.random.PRNGKey(777)
shape = jp.full((3,), 7)
sizes = jp.full((3,), 0.25)
x_center = -1.0
y_center = 1.0
centers = build_centers(shape, sizes, x_center, y_center)
scaling = jp.array(sizes) / 2.0
transforms = build_transform(centers, scaling)

lights = [
    paz.graphics.PointLight(
        jp.array([1.0, 1.0, 1.0]), jp.array([-10.0, 10.0, -10.0])
    )
]
world_to_camera = camera_pose = SE3.view_transform(
    jp.array([0, 4.0, -5.0]),
    jp.array([0.0, 0.0, 4.0]),
    jp.array([0.0, 1.0, 0.0]),
)
H, W = image_size = 224, 224
rays = paz.graphics.camera.build_rays(image_size, jp.pi / 3, camera_pose)

num_shapes = jp.prod(shape)
mask = jp.ones((num_shapes,), dtype=bool)
render = jax.jit(paz.graphics.Render((H, W), world_to_camera, rays))
shapes = build_shapes(key, num_shapes, transforms)

indices, neighbor_indices = get_neighbor_indices(shape)

for key in jax.random.split(key, 50):
    key, subkey = jax.random.split(key)
    mask = jax.random.randint(subkey, (num_shapes,), 0, 2)
    image, depth = render(shapes, mask, lights)
    loss = jax.jit(calculate_loss)(indices, neighbor_indices, mask)
    active_cubes = jp.sum(mask)
    print(f"\nTotal rendered cubes: {active_cubes}")
    print(f"Loss values: {loss}")
    plt.imshow(image)
    plt.title(f"Total Cubes: {active_cubes}")
    plt.show()
