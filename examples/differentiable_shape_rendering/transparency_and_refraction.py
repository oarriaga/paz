import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz

set_cache_dir(paz.logger.make_directory("cache"))


# --- Helper: Checkered Pattern Generation ---
def CheckeredImage(
    box_size=32,
    rows=16,
    cols=16,
    color_A=(0.0, 0.0, 0.0),
    color_B=(1.0, 1.0, 1.0),
):
    """Generates a black and white checkerboard image."""
    # Create indices grid
    indices = jp.indices((rows, cols)).sum(axis=0) % 2

    image_channels = []
    for i in range(3):
        # Create checker mask
        checkered = jp.kron(indices, jp.ones((box_size, box_size)))
        # Mix colors based on mask
        channel = color_A[i] * checkered + color_B[i] * (1 - checkered)
        image_channels.append(jp.expand_dims(channel, axis=-1))

    return jp.concatenate(image_channels, axis=-1)


# --- Materials ---

# [cite_start]1. Glass Material (Main Sphere) [cite: 829]
# High transparency and reflectivity for realistic glass with Fresnel.
glass_material = paz.graphics.Material(
    color=jp.array([0.0, 0.0, 0.1]),  # Very slight tint
    ambient=0.1,
    diffuse=0.1,
    specular=1.0,
    shininess=300.0,
    reflective=0.9,
    transparency=0.9,
    refractive_index=1.5,
)

# 2. Opaque Sphere Materials
# Shiny Blue (Left Sphere)
blue_material = paz.graphics.Material(
    color=jp.array([0.1, 0.1, 1.0]),
    ambient=0.1,
    diffuse=0.6,
    specular=0.3,
    shininess=100.0,
    reflective=0.2,  # Slight reflection
)

# Matte Red (Inside/Behind)
red_material = paz.graphics.Material(
    color=jp.array([0.8, 0.1, 0.1]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.0,
    shininess=10.0,
)

# Matte Green (Inside/Behind)
green_material = paz.graphics.Material(
    color=jp.array([0.1, 0.8, 0.1]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.0,
    shininess=10.0,
)

# Matte Dark Blue/Purple (Inside/Behind)
dark_blue_material = paz.graphics.Material(
    color=jp.array([0.1, 0.1, 0.6]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.0,
    shininess=10.0,
)

# 3. Wall Material (Checkered)
# Matte material to let the pattern do the work
wall_material = paz.graphics.Material(
    color=jp.array([1.0, 1.0, 1.0]),  # Base white, modulated by pattern
    ambient=0.2,
    diffuse=0.8,
    specular=0.0,
    shininess=10.0,
)

# Create the pattern
checkered_img = CheckeredImage()
# Scale the pattern transform so it repeats nicely on the walls
# Scaling by < 1 in the pattern transform makes the texture appear larger/less repeated?
# Usually Inverse transform is applied. Let's try scaling by 0.5 to make checks smaller (more repeats).
pattern_transform = paz.SE3.scaling(jp.full(3, 0.2))
wall_pattern = paz.graphics.PlanarPattern(
    checkered_img, transform=pattern_transform
)


# --- Scene Objects ---

# 1. The Room (Box of Planes)
room_radius = 10.0

# Floor (Y = 0)
floor = paz.graphics.Plane(
    paz.SE3.identity(), wall_material, wall_pattern  # Normal is +Y by default
)

# Ceiling (Y = 20)
ceiling = paz.graphics.Plane(
    paz.SE3.translation(jp.array([0.0, 20.0, 0.0])) @ paz.SE3.rotation_x(jp.pi),
    wall_material,
    wall_pattern,
)

# Back Wall (Z = 15)
back_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([0.0, 0.0, 15.0]))
    @ paz.SE3.rotation_x(jp.pi / 2),
    wall_material,
    wall_pattern,
)

# Left Wall (X = -15)
left_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([-15.0, 0.0, 0.0]))
    @ paz.SE3.rotation_z(-jp.pi / 2),
    wall_material,
    wall_pattern,
)

# Right Wall (X = 15)
right_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([15.0, 0.0, 0.0]))
    @ paz.SE3.rotation_z(jp.pi / 2),
    wall_material,
    wall_pattern,
)

# 2. Spheres

# Main Glass Sphere (Center)
# Radius approx 2.0 based on image
main_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([0.0, 2.0, 0.0]))
    @ paz.SE3.scaling(jp.full(3, 2.0)),
    glass_material,
)

# Blue Sphere (Left, Outside)
# Radius approx 1.0, positioned to the left and slightly back
left_blue_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([-3.5, 1.0, -1.0]))
    @ paz.SE3.scaling(jp.full(3, 1.0)),
    blue_material,
)

# Red Sphere (Behind Glass)
# Positioned behind the glass sphere so it refracts.
# Camera is at negative Z looking positive Z. Glass is at 0. Back objects are at +Z.
red_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([0.5, 1.5, 3.5]))
    @ paz.SE3.scaling(jp.full(3, 1.5)),
    red_material,
)

# Green Sphere (Behind Glass)
green_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([-1.5, 2.2, 3.0]))
    @ paz.SE3.scaling(jp.full(3, 0.7)),
    green_material,
)

# Dark Blue Sphere (Behind Glass)
dark_blue_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([2.2, 1.0, 3.0]))
    @ paz.SE3.scaling(jp.full(3, 0.7)),
    dark_blue_material,
)


scene = paz.graphics.Scene(
    [
        floor,
        ceiling,
        back_wall,
        left_wall,
        right_wall,
        main_sphere,
        left_blue_sphere,
        red_sphere,
        green_sphere,
        dark_blue_sphere,
    ]
)


# --- Camera & Lights ---

# Camera
# Positioned up and back to look down at the scene
camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 3.0, -8.0]),  # Eye
    jp.array([0.0, 1.0, 0.0]),  # Target
    jp.array([0.0, 1.0, 0.0]),  # Up
)

# Light
# Strong white light from top-left-front
lights = [
    paz.graphics.PointLight(jp.ones(3) * 0.9, jp.array([-10.0, 15.0, -10.0]))
]

# --- Rendering ---

H, W = 512, 512  # Good resolution for verification
y_FOV = jp.pi / 3.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)

render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        world_to_camera=camera_pose,
        rays=rays,
        lights=lights,
        shadows=True,  # Essential for the look
    )
)

print("Rendering Verification Scene...")
image, depth = render(scene=scene, mask=None)

# Post-processing
image = paz.image.denormalize(image)
paz.image.write("verification_scene.png", image)
paz.image.show(image)
