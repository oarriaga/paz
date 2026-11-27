import pytest
import jax.numpy as jp

from paz.graphics import albedo, cook_torrance
from paz.graphics.types import (
    PointLight,
    CookTorranceMaterial,
    Pattern,
    Shape,
)
from paz.graphics.constants import NO_PATTERN


@pytest.fixture
def simple_light():
    return PointLight(
        intensity=jp.array([1.0, 1.0, 1.0]),
        position=jp.array([0.0, 10.0, 0.0]),
    )


@pytest.fixture
def simple_pattern():
    dummy_image = jp.zeros((1, 1, 3))
    return Pattern(transform=jp.eye(4), type=NO_PATTERN, image=dummy_image)


@pytest.fixture
def simple_shape(simple_pattern):
    return Shape(
        transform=jp.eye(4), type=0, pattern=simple_pattern, material=None
    )


@pytest.fixture
def dielectric_material():
    return CookTorranceMaterial(
        color=jp.array([0.8, 0.2, 0.2]),
        ambient=0.1,
        base_reflectance=0.04,
        roughness=0.5,
        metallic=0.0,
    )


@pytest.fixture
def metallic_material():
    return CookTorranceMaterial(
        color=jp.array([0.9, 0.7, 0.4]),
        ambient=0.1,
        base_reflectance=0.04,
        roughness=0.3,
        metallic=1.0,
    )


def test_halfway_direction_is_unit_length():
    view = jp.array([[0.0, 1.0, 0.0]])
    light_direction = jp.array([[1.0, 0.0, 0.0]])
    halfway = cook_torrance.compute_halfway_direction(view, light_direction)
    norm = jp.linalg.norm(halfway, axis=-1)
    assert jp.allclose(norm, 1.0)


def test_microfacet_distribution_peaks_at_zero_angle():
    aligned = jp.array([1.0])
    grazing = jp.array([0.1])
    aligned_value = cook_torrance.compute_microfacet_distribution(aligned, 0.3)
    grazing_value = cook_torrance.compute_microfacet_distribution(grazing, 0.3)
    assert aligned_value > grazing_value


def test_microfacet_distribution_spreads_with_higher_roughness():
    aligned = jp.array([1.0])
    smooth = cook_torrance.compute_microfacet_distribution(aligned, 0.1)
    rough = cook_torrance.compute_microfacet_distribution(aligned, 0.9)
    assert smooth > rough


def test_visibility_is_in_unit_range():
    cosines = jp.linspace(0.05, 1.0, 8)
    values = cook_torrance.compute_visibility(cosines, 0.5)
    assert jp.all(values >= 0.0)
    assert jp.all(values <= 1.0)


def test_geometry_term_is_in_unit_range():
    normal_dot_light = jp.array([0.5, 0.9])
    normal_dot_view = jp.array([0.4, 0.8])
    values = cook_torrance.compute_geometry_term(
        normal_dot_light, normal_dot_view, 0.5
    )
    assert jp.all(values >= 0.0)
    assert jp.all(values <= 1.0)


def test_fresnel_reflectance_at_normal_incidence_returns_base():
    base_reflectance = jp.array([0.04, 0.04, 0.04])
    view_dot_half = jp.array([1.0])
    reflectance = cook_torrance.compute_fresnel_reflectance(
        view_dot_half, base_reflectance
    )
    assert jp.allclose(reflectance, base_reflectance)


def test_fresnel_reflectance_at_grazing_angle_returns_one():
    base_reflectance = jp.array([0.04, 0.04, 0.04])
    view_dot_half = jp.array([0.0])
    reflectance = cook_torrance.compute_fresnel_reflectance(
        view_dot_half, base_reflectance
    )
    assert jp.allclose(reflectance, jp.ones(3))


def test_metallic_base_reflectance_matches_albedo(metallic_material):
    base_reflectance = cook_torrance.compute_base_reflectance(metallic_material)
    assert jp.allclose(base_reflectance, metallic_material.color)


def test_dielectric_base_reflectance_is_constant(dielectric_material):
    base_reflectance = cook_torrance.compute_base_reflectance(
        dielectric_material
    )
    expected = jp.full_like(
        dielectric_material.color, dielectric_material.base_reflectance
    )
    assert jp.allclose(base_reflectance, expected)


def test_diffuse_color_vanishes_for_metallic_surfaces(metallic_material):
    albedo = jp.array([[0.5, 0.5, 0.5]])
    reflectance = jp.array([[0.04, 0.04, 0.04]])
    diffuse = cook_torrance.compute_diffuse_color(
        metallic_material, albedo, reflectance
    )
    assert jp.allclose(diffuse, jp.zeros_like(diffuse))


def test_compute_colors_runs_with_dielectric(
    simple_shape, dielectric_material, simple_light
):
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])
    args = simple_shape, dielectric_material, points
    shape_albedo = albedo.compute_shape_albedo(*args)
    colors = cook_torrance.compute_colors(
        shape_albedo, dielectric_material, points, normals, eye, simple_light
    )
    assert colors.shape == (1, 3)
    assert jp.all(colors >= 0.0)


def test_compute_colors_is_zero_for_back_facing_normals(
    simple_shape, dielectric_material, simple_light
):
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, -1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])
    args = simple_shape, dielectric_material, points
    shape_albedo = albedo.compute_shape_albedo(*args)
    colors = cook_torrance.compute_colors(
        shape_albedo, dielectric_material, points, normals, eye, simple_light
    )
    base_color = dielectric_material.color * simple_light.intensity
    expected_ambient = base_color * dielectric_material.ambient
    assert jp.allclose(colors, expected_ambient)


def test_smoother_surfaces_concentrate_specular(
    simple_shape, simple_light
):
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])
    smooth = CookTorranceMaterial(
        color=jp.zeros(3), ambient=0.0, base_reflectance=0.04,
        roughness=0.05, metallic=0.0,
    )
    rough = smooth._replace(roughness=0.9)
    shape_albedo = albedo.compute_shape_albedo(simple_shape, smooth, points)
    smooth_colors = cook_torrance.compute_colors(
        shape_albedo, smooth, points, normals, eye, simple_light
    )
    rough_colors = cook_torrance.compute_colors(
        shape_albedo, rough, points, normals, eye, simple_light
    )
    assert jp.sum(smooth_colors) > jp.sum(rough_colors)


def test_compute_colors_with_shadow_uses_ambient(
    simple_shape, dielectric_material, simple_light
):
    points = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])
    is_shadow = jp.array([1.0])
    args = simple_shape, dielectric_material, points
    shape_albedo = albedo.compute_shape_albedo(*args)
    colors = cook_torrance.compute_colors_with_shadow(
        shape_albedo, dielectric_material, points, normals, eye,
        simple_light, is_shadow,
    )
    expected = cook_torrance.compute_ambient(
        shape_albedo, dielectric_material, simple_light
    )
    assert jp.allclose(colors, expected)


@pytest.mark.parametrize("metallic", [0.0, 1.0])
def test_compute_colors_scales_linearly_with_intensity(metallic):
    material = CookTorranceMaterial(jp.full(3, 0.5), 0.0, 0.04, 0.9, metallic)
    intensities = [0.5, 1.0, 2.0]
    colors = [compute_head_on_colors(material, value) for value in intensities]
    assert jp.allclose(colors[1] / colors[0], 2.0, atol=1e-6)
    assert jp.allclose(colors[2] / colors[1], 2.0, atol=1e-6)


def compute_head_on_colors(material, intensity):
    points = jp.zeros((1, 3))
    normals = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, 0.0]])
    position = jp.array([0.0, 1.0, 0.0])
    light = PointLight(jp.full(3, intensity), position)
    args = material.color[None], material, points, normals, eye, light
    return cook_torrance.compute_colors(*args)


def test_diffuse_matches_lambertian_by_hand():
    material = CookTorranceMaterial(jp.full(3, 0.5), 0.0, 0.04, 1.0, 0.0)
    black_material = material._replace(color=jp.zeros(3))
    intensity = 2.0
    colors = compute_head_on_colors(material, intensity)
    specular = compute_head_on_colors(black_material, intensity)
    expected = 0.5 / jp.pi * (1.0 - 0.04) * (1.0 - 0.0) * 1.0 * intensity
    assert jp.allclose(colors - specular, expected, atol=1e-6)
