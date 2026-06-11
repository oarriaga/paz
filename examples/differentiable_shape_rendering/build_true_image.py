import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
from functools import partial

import jax
import jax.numpy as jp

import paz

# jax.config.update("jax_platform_name", "cpu")

TEXTURE_SIZE = 768
world_up = jp.array([0.0, 1.0, 0.0])
light_position = jp.array([2.4, 3.8, -2.0])
lights = paz.graphics.PointLight(jp.full(3, 1.32), light_position)

camera_shift = 1.8 * jp.array([-1.0, 1.0, -2.0])
world_to_camera = paz.SE3.view_transform(camera_shift, jp.zeros(3), world_up)

H, W, y_FOV, resize_factor = image_size = 2 * 1024, 2 * 1024, jp.pi / 5.0, 2
render_kwargs = {
    "mask": None,
    "shadows": True,
    "lights": lights,
    "tiles": (1, 1),
    "chunk_size": 2**10,
}
rays = paz.graphics.camera.build_rays(image_size, y_FOV, world_to_camera)
# render_args = ((H, W), world_to_camera, rays)
render_args = ((H, W), y_FOV, world_to_camera)


def make_uv_grid(height, width):
    u = jp.linspace(0.0, 1.0, width)[None, :]
    v = jp.linspace(0.0, 1.0, height)[:, None]
    u = jp.broadcast_to(u, (height, width))
    v = jp.broadcast_to(v, (height, width))
    return u, v


def normalize_map(values):
    min_value = jp.min(values)
    max_value = jp.max(values)
    return (values - min_value) / (max_value - min_value + 1e-6)


def make_limestone_floor_texture(
    height=TEXTURE_SIZE, width=TEXTURE_SIZE, slabs_x=6, slabs_y=4
):
    u, v = make_uv_grid(height, width)
    slab_u = u * slabs_x
    slab_v = v * slabs_y
    slab_u_id = jp.floor(slab_u)
    slab_v_id = jp.floor(slab_v)
    slab_u_local = slab_u - slab_u_id
    slab_v_local = slab_v - slab_v_id
    grout_width = 0.024
    grout = (
        (slab_u_local < grout_width)
        | (slab_u_local > (1.0 - grout_width))
        | (slab_v_local < grout_width)
        | (slab_v_local > (1.0 - grout_width))
    )
    grout = grout.astype(jp.float32)

    macro = jp.sin(2.0 * jp.pi * (1.5 * u + 1.2 * v))
    veins = jp.sin(
        2.0 * jp.pi * (5.8 * u - 3.7 * v + 0.45 * jp.sin(2.0 * jp.pi * 1.3 * v))
    )
    micro = jp.sin(2.0 * jp.pi * (18.0 * u + 14.0 * v))
    mineral = normalize_map(0.55 * macro + 0.35 * veins + 0.10 * micro)
    slab_tint = 0.95 + 0.08 * jp.sin(1.9 * slab_u_id + 1.4 * slab_v_id + 0.3)
    stone = jp.clip(
        0.53 + 0.16 * (mineral - 0.5) + 0.08 * (slab_tint - 1.0), 0.0, 1.0
    )
    tone = jp.where(grout > 0.0, 0.40 + 0.03 * mineral, stone)

    red = 0.50 + 0.22 * tone
    green = 0.49 + 0.20 * tone
    blue = 0.45 + 0.18 * tone
    return jp.clip(jp.stack([red, green, blue], axis=-1), 0.0, 1.0)


def make_marble_texture(height=TEXTURE_SIZE, width=TEXTURE_SIZE):
    u, v = make_uv_grid(height, width)
    clouds_low = jp.sin(2.0 * jp.pi * (1.8 * u - 1.4 * v))
    clouds_mid = jp.sin(2.0 * jp.pi * (4.0 * u + 3.4 * v))
    clouds = normalize_map(0.62 * clouds_low + 0.38 * clouds_mid)
    vein_basis = jp.sin(2.0 * jp.pi * (11.0 * u + 7.0 * v + 0.35 * clouds_low))
    veins = jp.power(1.0 - jp.abs(vein_basis), 4.0)
    marble = jp.clip(0.60 + 0.20 * (clouds - 0.5) + 0.18 * veins, 0.0, 1.0)

    red = 0.40 + 0.23 * marble
    green = 0.44 + 0.25 * marble
    blue = 0.50 + 0.30 * marble
    return jp.clip(jp.stack([red, green, blue], axis=-1), 0.0, 1.0)


def make_terracotta_texture(height=TEXTURE_SIZE, width=TEXTURE_SIZE):
    u, v = make_uv_grid(height, width)
    mottling_a = jp.sin(2.0 * jp.pi * (4.8 * u + 3.2 * v))
    mottling_b = jp.sin(2.0 * jp.pi * (10.5 * v - 2.4 * u))
    speckles = jp.sin(2.0 * jp.pi * (21.0 * u + 17.0 * v))
    micro = jp.power(0.5 + 0.5 * speckles, 6.0)
    terracotta = normalize_map(
        0.60 * mottling_a + 0.30 * mottling_b + 0.10 * micro
    )
    tone = jp.clip(0.54 + 0.18 * (terracotta - 0.5), 0.0, 1.0)

    red = 0.42 + 0.22 * tone
    green = 0.24 + 0.13 * tone
    blue = 0.18 + 0.08 * tone
    return jp.clip(jp.stack([red, green, blue], axis=-1), 0.0, 1.0)


# floor ------------------------------------------------
floor_texture = make_limestone_floor_texture()
floor_pattern_transform = paz.SE3.translation(jp.array([1.25, 0.0, 1.25])) @ paz.SE3.scaling(jp.full(3, 2.5))  # fmt: skip
floor_pattern = paz.graphics.PlanarPattern(
    floor_texture, floor_pattern_transform
)
floor_half_height = 0.1
floor_size = paz.SE3.scaling(jp.array([1.5, floor_half_height, 1.5]))
floor_material = paz.graphics.Material(
    color=jp.array([0.03, 0.028, 0.022]),
    ambient=0.08,
    diffuse=0.70,
    specular=0.11,
    shininess=76.0,
)
floor = paz.graphics.Cube(floor_size, floor_material, floor_pattern)
floor = floor._replace(transform=floor_size)

# shape 0 (sphere) -------------------------------------
sphere_texture = make_marble_texture()
sphere_pattern = paz.graphics.SphericalPattern(
    sphere_texture, paz.SE3.rotation_z(jp.pi / 2.0)
)
sphere_scale = paz.SE3.scaling(sphere_radius := 0.35)
sphere_y = floor_half_height + sphere_radius + 0.002
sphere_shift = paz.SE3.translation(jp.array([1.00, sphere_y, 0.7]))
sphere_transform = sphere_shift @ sphere_scale
sphere_material = paz.graphics.Material(
    color=jp.array([0.01, 0.02, 0.03]),
    ambient=0.09,
    diffuse=0.64,
    specular=0.42,
    shininess=245.0,
)
sphere = paz.graphics.Sphere(sphere_transform, sphere_material, sphere_pattern)

# shape 1 (cone) ---------------------------------------
cone_texture = make_terracotta_texture()
cone_pattern = paz.graphics.PlanarPattern(
    cone_texture, paz.SE3.scaling(jp.full(3, 1.4))
)
cone_scale = paz.SE3.scaling(cone_size := 0.45)
cone_shift = paz.SE3.translation(jp.array([-0.6, cone_size, -0.4]))
cone_material = paz.graphics.Material(
    color=jp.array([0.03, 0.018, 0.014]),
    ambient=0.09,
    diffuse=0.66,
    specular=0.24,
    shininess=145.0,
)
cone = paz.graphics.Cone(cone_shift @ cone_scale, cone_material, cone_pattern)

shapes = [floor, sphere, cone]
scene = paz.graphics.Scene(shapes)
# paz.graphics.render(shape, y_FOV, pose, scene)
render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))
image, _ = render(scene=scene)
image = paz.image.denormalize(image)
image = paz.image.resize_opencv(image, (H // resize_factor, W // resize_factor))
paz.image.show(image)
paz.image.write("true_image.png", image)
