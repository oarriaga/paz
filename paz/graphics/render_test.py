import pytest
import jax.numpy as jp

from paz.graphics import shapes
from paz.graphics import render
from paz import algebra
from paz.graphics.types import PointLight, Material, Shape, Pattern
from paz.graphics.constants import NO_PATTERN, SPHERE, PLANE, WHITE, RED, BLACK


@pytest.fixture
def simple_light():
    """A single white light source from above and behind the camera."""
    return PointLight(intensity=WHITE, position=jp.array([0.0, 10.0, -10.0]))


@pytest.fixture
def red_sphere_material():
    """A material for a matte red sphere."""
    return Material(
        color=RED, ambient=0.1, diffuse=0.9, specular=0.1, shininess=200.0
    )


@pytest.fixture
def white_floor_material():
    """A material for a matte white floor."""
    return Material(
        color=WHITE, ambient=0.1, diffuse=0.9, specular=0.1, shininess=200.0
    )


@pytest.fixture
def default_pattern():
    """A default pattern with a dummy image."""
    dummy_image = jp.zeros((1, 1, 3))
    return Pattern(transform=jp.eye(4), type=NO_PATTERN, image=dummy_image)


@pytest.fixture
def red_sphere(red_sphere_material, default_pattern):
    """A unit sphere translated up by 1 unit."""
    transform = jp.eye(4).at[1, 3].set(1.0)
    return Shape(
        transform=transform,
        type=SPHERE,
        pattern=default_pattern,
        material=red_sphere_material,
    )


@pytest.fixture
def floor_plane(white_floor_material, default_pattern):
    """A unit plane at the origin, acting as a floor."""
    return Shape(
        transform=jp.eye(4),
        type=PLANE,
        pattern=default_pattern,
        material=white_floor_material,
    )


@pytest.fixture
def simple_scene(red_sphere, floor_plane):
    """A scene with a red sphere sitting on a white floor."""
    return shapes.merge(red_sphere, floor_plane)


def test_select_colors():
    """Tests if the function selects colors from the closest object."""
    depths = jp.array([[10.0, 5.0], [20.0, 3.0]])
    depths = depths[:, :, jp.newaxis]
    colors = jp.array([[RED, RED], [WHITE, WHITE]])
    selected = render.select_colors(depths, colors)
    expected = jp.vstack([RED, WHITE])
    assert jp.allclose(selected, expected)


def test_invert_inside_normals():
    """Tests that normals facing away from the camera are flipped."""
    eye = jp.array([[0.0, 0.0, -1.0]])
    normal_facing_away = jp.array([[0.0, 0.0, 1.0]])
    inverted = render.invert_inside_normals(eye, normal_facing_away)
    expected = jp.array([[0.0, 0.0, -1.0]])
    assert jp.allclose(inverted, expected)


def test_compute_scene_hit_mask():
    """Tests that hit masks from multiple objects are combined correctly."""
    masks = jp.array([[True, False], [False, True]])
    combined = render.compute_scene_hit_mask(masks)
    expected = jp.array([True, True])
    assert jp.all(combined == expected)


def test_to_color_image_reshaping():
    """Tests the reshaping and data type logic of the final image conversion."""
    hit_mask = jp.ones((4), dtype=bool)
    colors = jp.array([RED, WHITE, RED, WHITE]) * 0.5
    image = render.to_color_image(hit_mask, colors, (2, 2))
    assert image.shape == (2, 2, 3)
    assert image.dtype == jp.float32


def test_render_without_shadows(simple_scene, simple_light):
    """Tests the main render function with a simple scene and no shadows."""
    image_shape = (1, 2)
    H, W = image_shape
    camera_transform = jp.eye(4).at[2, 3].set(-5.0)
    world_to_camera = jp.linalg.inv(camera_transform)

    rays = (
        jp.array([[0, 0, -5], [0, 0, -5]]),
        jp.array([[0, -0.5, 1], [0, 0, 1]]),
    )

    render_function = render.Render(
        image_shape, world_to_camera, rays, shadows=False
    )
    image, depth = render_function(simple_scene, jp.ones(2), [simple_light])

    assert image.shape == (H, W, 3)
    assert depth.shape == (H, W)
    assert not jp.allclose(image[0, 0], BLACK)
    assert not jp.allclose(image[0, 1], BLACK)


def test_render_with_shadows(simple_scene):
    """Tests that a shadow is correctly cast from one object to another."""
    image_shape = (1, 2)
    H, W = image_shape
    light = PointLight(intensity=WHITE, position=jp.array([0.0, 5.0, 0.0]))

    # FIX: Set up a camera that is guaranteed to see the shadow
    camera_origin = jp.array([0.0, 2.0, -5.0])
    camera_transform = jp.eye(4).at[:3, 3].set(camera_origin)
    world_to_camera = jp.linalg.inv(camera_transform)

    # FIX: Define rays that hit the floor inside and outside the shadow
    shadow_target = jp.array([0.0, 0.0, 0.0])
    lit_target = jp.array([2.0, 0.0, 0.0])

    shadow_ray_dir = algebra.normalize(shadow_target - camera_origin)
    lit_ray_dir = algebra.normalize(lit_target - camera_origin)

    rays = (
        jp.vstack([camera_origin, camera_origin]),
        jp.vstack([shadow_ray_dir, lit_ray_dir]),
    )

    render_function = render.Render(
        image_shape, world_to_camera, rays, shadows=True
    )
    image, _ = render_function(simple_scene, jp.ones(2), [light])

    pixel_in_shadow = image[0, 0]
    pixel_in_light = image[0, 1]

    assert jp.linalg.norm(pixel_in_shadow) < jp.linalg.norm(pixel_in_light)
