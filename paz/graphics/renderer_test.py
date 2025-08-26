import pytest
import jax.numpy as jp

from paz.graphics import shapes
from paz.graphics import renderer
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
    """A scene with a red sphere and a white floor, as a list."""
    return [red_sphere, floor_plane]


def test_select_colors():
    """Tests if the function selects colors from the closest object."""
    depths = jp.array([[10.0, 5.0], [20.0, 3.0]])
    depths = depths[:, :, jp.newaxis]
    colors = jp.array([[RED, RED], [WHITE, WHITE]])
    selected = renderer.select_colors(depths, colors)
    expected = jp.vstack([RED, WHITE])
    assert jp.allclose(selected, expected)


def test_invert_inside_normals():
    """Tests that normals facing away from the camera are flipped."""
    eye = jp.array([[0.0, 0.0, -1.0]])
    normal_facing_away = jp.array([[0.0, 0.0, 1.0]])
    inverted = renderer.invert_inside_normals(eye, normal_facing_away)
    expected = jp.array([[0.0, 0.0, -1.0]])
    assert jp.allclose(inverted, expected)


def test_compute_scene_hit_mask():
    """Tests that hit masks from multiple objects are combined correctly."""
    masks = jp.array([[True, False], [False, True]])
    combined = renderer.compute_scene_hit_mask(masks)
    expected = jp.array([True, True])
    assert jp.all(combined == expected)


def test_to_color_image_reshaping():
    """Tests the reshaping and data type logic of the final image conversion."""
    hit_mask = jp.ones((4), dtype=bool)
    colors = jp.array([RED, WHITE, RED, WHITE]) * 0.5
    image = renderer.to_color_image(hit_mask, colors, (2, 2))
    assert image.shape == (2, 2, 3)
    assert image.dtype == jp.float32


def test_prepare_lights_with_single_light(simple_light):
    """Tests that a single PointLight is correctly wrapped in a list."""
    processed = renderer.prepare_lights(simple_light)
    assert isinstance(processed, list)
    assert len(processed) == 1
    assert processed[0] == simple_light


def test_prepare_lights_with_list(simple_light):
    """Tests that a list of PointLights is returned unchanged."""
    light_list = [simple_light, simple_light]
    processed = renderer.prepare_lights(light_list)
    assert processed == light_list


def test_prepare_lights_with_invalid_type():
    """Tests that a non-list, non-PointLight input raises a TypeError."""
    with pytest.raises(TypeError):
        renderer.prepare_lights("not_a_light")


def test_prepare_lights_with_invalid_list_contents(simple_light):
    """Tests that a list with invalid contents raises a TypeError."""
    with pytest.raises(TypeError):
        renderer.prepare_lights([simple_light, "not_a_light"])


def test_prepare_shapes_with_single_shape(red_sphere):
    """Tests that a single Shape is expanded to a batched Shape."""
    processed = renderer.prepare_shapes(red_sphere)
    assert isinstance(processed, Shape)
    assert processed.transform.shape[0] == 1


def test_prepare_shapes_with_list(simple_scene):
    """Tests that a list of Shapes is merged into a single batched Shape."""
    processed = renderer.prepare_shapes(simple_scene)
    assert isinstance(processed, Shape)
    assert processed.transform.shape[0] == len(simple_scene)


def test_prepare_shapes_with_invalid_type():
    """Tests that an invalid input type raises a TypeError."""
    with pytest.raises(TypeError):
        renderer.prepare_shapes(123)


def test_prepare_shapes_with_invalid_list_contents(red_sphere):
    """Tests that a list with invalid contents raises a TypeError."""
    with pytest.raises(TypeError):
        renderer.prepare_shapes([red_sphere, 123])


def test_render_without_shadows(simple_scene, simple_light):
    """Tests the main renderer function using the flexible API."""
    image_shape = (1, 2)
    H, W = image_shape
    camera_transform = jp.eye(4).at[2, 3].set(-5.0)
    world_to_camera = jp.linalg.inv(camera_transform)
    rays = (
        jp.array([[0, 0, -5], [0, 0, -5]]),
        jp.array([[0, -0.5, 1], [0, 0, 1]]),
    )

    # Call the renderer function without the mask, passing the list directly
    image, depth = renderer.render(
        image_shape, world_to_camera, rays, simple_scene, [simple_light]
    )

    assert image.shape == (H, W, 3)
    assert depth.shape == (H, W)
    assert not jp.allclose(image[0, 0], BLACK)
    assert not jp.allclose(image[0, 1], BLACK)


def test_render_with_shadows(simple_scene):
    """Tests the shadow renderer using the flexible API."""
    image_shape = (1, 2)
    H, W = image_shape
    light = PointLight(intensity=WHITE, position=jp.array([0.0, 5.0, 0.0]))
    camera_origin = jp.array([0.0, 2.0, -5.0])
    camera_transform = jp.eye(4).at[:3, 3].set(camera_origin)
    world_to_camera = jp.linalg.inv(camera_transform)

    shadow_target = jp.array([0.0, 0.0, 0.0])
    lit_target = jp.array([2.0, 0.0, 0.0])

    shadow_ray_dir = algebra.normalize(shadow_target - camera_origin)
    lit_ray_dir = algebra.normalize(lit_target - camera_origin)

    rays = (
        jp.vstack([camera_origin, camera_origin]),
        jp.vstack([shadow_ray_dir, lit_ray_dir]),
    )

    # Call the shadow renderer without the mask, passing the list directly
    image, _ = renderer.render_with_shadows(
        image_shape, world_to_camera, rays, simple_scene, [light]
    )

    pixel_in_shadow = image[0, 0]
    pixel_in_light = image[0, 1]

    assert jp.linalg.norm(pixel_in_shadow) < jp.linalg.norm(pixel_in_light)
