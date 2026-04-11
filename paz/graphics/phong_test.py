import pytest
import jax.numpy as jp

from paz.graphics import phong
from paz import algebra
from paz.graphics.types import PointLight, Material, Pattern, Shape
from paz.graphics.constants import NO_PATTERN, PLANAR_PATTERN


@pytest.fixture
def simple_light():
    """A simple white light source."""
    return PointLight(
        intensity=jp.array([1.0, 1.0, 1.0]),
        position=jp.array([0.0, 10.0, -10.0]),
    )


@pytest.fixture
def simple_material():
    """A standard, non-patterned material."""
    return Material(
        color=jp.array([1.0, 0.2, 0.2]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.5,
        shininess=200.0,
    )


@pytest.fixture
def simple_shape(simple_pattern):
    """A simple shape with a basic pattern."""
    return Shape(
        transform=jp.eye(4), type=0, pattern=simple_pattern, material=None
    )


@pytest.fixture
def simple_pattern():
    """A default, empty pattern with a dummy image."""
    dummy_image = jp.zeros((1, 1, 3))
    return Pattern(transform=jp.eye(4), type=NO_PATTERN, image=dummy_image)


def test_compute_colors_in_shape_with_transforms():
    """Tests the full world-to-pattern coordinate transformation."""
    points_world = jp.array([[11.0, 2.0, 3.0]])
    shape_transform = (
        jp.eye(4).at[0, 3].set(10.0)
    )  # Translate shape by +10 on X
    pattern_transform = jp.eye(4).at[0, 0].set(2.0)  # Scale pattern by 2x on X

    # World -> Shape: (11, 2, 3) -> (1, 2, 3)
    # Shape -> Pattern: (1, 2, 3) -> (0.5, 2, 3)
    expected_points = jp.array([[0.5, 2.0, 3.0]])

    points_pattern = phong.compute_colors_in_shape(
        pattern_transform, shape_transform, points_world
    )
    assert jp.allclose(points_pattern, expected_points)


def test_compute_base_color_scales_pattern_with_light_intensity():
    pattern_image = jp.broadcast_to(jp.array([0.2, 0.3, 0.1]), (2, 2, 3))
    pattern = Pattern(jp.eye(4), PLANAR_PATTERN, pattern_image)
    shape = Shape(jp.eye(4), 0, None, pattern)
    material = Material(color=jp.array([0.4, 0.1, 0.2]))
    light = PointLight(
        intensity=jp.array([0.5, 0.25, 0.8]),
        position=jp.array([0.0, 1.0, 0.0]),
    )
    points = jp.array([[0.0, 0.0, 0.0]])
    expected_color = (jp.array([[0.2, 0.3, 0.1]]) + material.color)
    expected_color = expected_color * light.intensity
    color = phong.compute_base_color(shape, material, light, points)
    assert jp.allclose(color, expected_color)


def test_compute_base_color_with_pattern_is_zero_for_zero_light():
    pattern_image = jp.full((2, 2, 3), 0.5)
    pattern = Pattern(jp.eye(4), PLANAR_PATTERN, pattern_image)
    shape = Shape(jp.eye(4), 0, None, pattern)
    material = Material(color=jp.array([0.4, 0.1, 0.2]))
    light = PointLight(
        intensity=jp.zeros(3),
        position=jp.array([0.0, 1.0, 0.0]),
    )
    points = jp.array([[0.0, 0.0, 0.0]])
    color = phong.compute_base_color(shape, material, light, points)
    assert jp.allclose(color, jp.zeros((1, 3)))


def test_compute_ambient(simple_shape, simple_material, simple_light):
    """Tests the ambient light calculation."""
    points = jp.array([[0.0, 0.0, 0.0]])
    expected_base = simple_material.color * simple_light.intensity
    expected_ambient = expected_base * simple_material.ambient

    ambient = phong.compute_ambient(
        simple_shape, simple_material, simple_light, points
    )
    assert jp.allclose(ambient, expected_ambient)


def test_compute_diffuse_with_direct_light(
    simple_shape, simple_material, simple_light
):
    """Tests diffuse component when light is perpendicular to the surface."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_pos_direct = jp.array([0.0, 10.0, 0.0])
    light = simple_light._replace(position=light_pos_direct)

    base_color = simple_material.color * light.intensity
    expected_diffuse = base_color * simple_material.diffuse

    diffuse = phong.compute_diffuse(
        simple_shape, simple_material, light, points, normals
    )
    assert jp.allclose(diffuse, expected_diffuse)


def test_compute_diffuse_with_45_degree_light(
    simple_shape, simple_material, simple_light
):
    """Tests diffuse component when light is at a 45-degree angle."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_pos_45_deg = jp.array([10.0, 10.0, 0.0])
    light = simple_light._replace(position=light_pos_45_deg)

    # cos(45 degrees) is approx 0.7071
    cos_theta = jp.sqrt(2) / 2
    base_color = simple_material.color * light.intensity
    expected_diffuse = base_color * simple_material.diffuse * cos_theta

    diffuse = phong.compute_diffuse(
        simple_shape, simple_material, light, points, normals
    )
    assert jp.allclose(diffuse, expected_diffuse)


def test_compute_diffuse_with_grazing_light(
    simple_shape, simple_material, simple_light
):
    """Tests diffuse component is zero when light is at a 90-degree angle."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_pos_grazing = jp.array([10.0, 0.0, 0.0])
    light = simple_light._replace(position=light_pos_grazing)

    expected_diffuse = jp.array([0.0, 0.0, 0.0])

    diffuse = phong.compute_diffuse(
        simple_shape, simple_material, light, points, normals
    )
    assert jp.allclose(diffuse, expected_diffuse)


def test_compute_diffuse_with_light_behind_surface(
    simple_shape, simple_material, simple_light
):
    """Tests diffuse component is zero when light is behind the surface."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_pos_behind = jp.array([0.0, -10.0, 0.0])  # Behind the normal
    light = simple_light._replace(position=light_pos_behind)

    # Dot product will be negative, clamped to 0
    expected_diffuse = jp.array([0.0, 0.0, 0.0])

    diffuse = phong.compute_diffuse(
        simple_shape, simple_material, light, points, normals
    )
    assert jp.allclose(diffuse, expected_diffuse)


def test_compute_specular_with_perfect_reflection(
    simple_material, simple_light
):
    """Tests specular highlight when eye is aligned with reflection vector."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    light_pos = jp.array([0.0, 10.0, -10.0])
    light = simple_light._replace(position=light_pos)
    eye = algebra.normalize(jp.array([[0.0, -10.0, -10.0]]))

    expected_specular = light.intensity * simple_material.specular

    specular = phong.compute_specular(
        simple_material, light, points, normals, eye
    )
    assert jp.all(specular <= expected_specular + 1e-6)
    assert jp.allclose(specular, expected_specular, atol=2e-3)


def test_compute_specular_normalizes_non_unit_eye(
    simple_material, simple_light
):
    """Tests specular stays bounded for a scaled eye vector."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    light_pos = jp.array([0.0, 10.0, -10.0])
    light = simple_light._replace(position=light_pos)
    eye = jp.array([[0.0, -10.0, -10.0]])

    expected_specular = light.intensity * simple_material.specular
    specular = phong.compute_specular(
        simple_material, light, points, normals, eye
    )

    assert jp.all(specular <= expected_specular + 1e-6)
    assert jp.allclose(specular, expected_specular, atol=1e-3)


def test_compute_specular_with_no_reflection(simple_material, simple_light):
    """Tests specular highlight is zero when eye is away from reflection."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])

    expected_specular = jp.array([0.0, 0.0, 0.0])

    specular = phong.compute_specular(
        simple_material, simple_light, points, normals, eye
    )
    assert jp.allclose(specular, expected_specular, atol=1e-6)


def test_compute_specular_effect_of_shininess(simple_light):
    """Tests that a higher shininess value creates a smaller highlight."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    # Eye is slightly off the perfect reflection path
    eye = algebra.normalize(jp.array([[0.1, 0.0, -1.0]]))

    material_high_shine = Material(
        color=jp.ones(3), ambient=0, diffuse=0, specular=1.0, shininess=200.0
    )
    material_low_shine = Material(
        color=jp.ones(3), ambient=0, diffuse=0, specular=1.0, shininess=10.0
    )

    specular_high = phong.compute_specular(
        material_high_shine, simple_light, points, normals, eye
    )
    specular_low = phong.compute_specular(
        material_low_shine, simple_light, points, normals, eye
    )

    # Assert that the high-shininess highlight is closer to zero
    assert jp.linalg.norm(specular_high) < jp.linalg.norm(specular_low)


def test_compute_specular_shininess_with_non_unit_eye(simple_light):
    """Tests shininess falls off for a scaled off-axis eye vector."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[1.0, 0.0, -10.0]])

    material_high_shine = Material(
        color=jp.ones(3), ambient=0, diffuse=0, specular=1.0, shininess=200.0
    )
    material_low_shine = Material(
        color=jp.ones(3), ambient=0, diffuse=0, specular=1.0, shininess=10.0
    )

    specular_high = phong.compute_specular(
        material_high_shine, simple_light, points, normals, eye
    )
    specular_low = phong.compute_specular(
        material_low_shine, simple_light, points, normals, eye
    )

    expected_high = simple_light.intensity * material_high_shine.specular
    expected_low = simple_light.intensity * material_low_shine.specular

    assert jp.all(specular_high <= expected_high + 1e-6)
    assert jp.all(specular_low <= expected_low + 1e-6)
    assert jp.linalg.norm(specular_high) < jp.linalg.norm(specular_low)


def test_compute_colors_is_sum_of_components(
    simple_shape, simple_material, simple_light
):
    """Tests that final color is the sum of ambient, diffuse, and specular."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])

    # Manually calculate each component
    ambient = phong.compute_ambient(
        simple_shape, simple_material, simple_light, points
    )
    diffuse = phong.compute_diffuse(
        simple_shape, simple_material, simple_light, points, normals
    )
    specular = phong.compute_specular(
        simple_material, simple_light, points, normals, eye
    )
    expected_color = ambient + diffuse + specular

    # Calculate with the main function
    actual_color = phong.compute_colors(
        simple_shape, simple_material, points, normals, eye, simple_light
    )

    assert jp.allclose(actual_color, expected_color, 1e-3)


def test_compute_colors_with_shadow_is_ambient(
    simple_shape, simple_material, simple_light
):
    """Tests that a shadowed color is only the ambient component."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])
    shadow_mask = jp.array([True])

    expected_ambient = phong.compute_ambient(
        simple_shape, simple_material, simple_light, points
    )
    shadowed_color = phong.compute_colors_with_shadow(
        simple_shape,
        simple_material,
        points,
        normals,
        eye,
        simple_light,
        shadow_mask,
    )

    assert jp.allclose(shadowed_color, expected_ambient)


def test_compute_colors_with_no_shadow_is_full_color(
    simple_shape, simple_material, simple_light
):
    """Tests that a non-shadowed color is the full Phong calculation."""
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])
    shadow_mask = jp.array([False])

    expected_full_color = phong.compute_colors(
        simple_shape, simple_material, points, normals, eye, simple_light
    )
    non_shadowed_color = phong.compute_colors_with_shadow(
        simple_shape,
        simple_material,
        points,
        normals,
        eye,
        simple_light,
        shadow_mask,
    )

    assert jp.allclose(non_shadowed_color, expected_full_color)
