import pytest
import jax.numpy as jp
import paz
from paz.graphics.types import Material, Shape, Pattern


@pytest.fixture
def material():
    """Provides a simple, shared Material object."""
    return Material()


@pytest.fixture
def pattern_256():
    """Provides a Pattern with a 256x256 image."""
    return Pattern(image=jp.zeros((256, 256, 3)))


@pytest.fixture
def pattern_512():
    """Provides a Pattern with a 512x512 image."""
    return Pattern(image=jp.zeros((512, 512, 3)))


@pytest.fixture
def shape_256_A(material, pattern_256):
    """Provides a shape with a 256x256 pattern."""
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.SPHERE,
        material=material,
        pattern=pattern_256,
    )


@pytest.fixture
def shape_256_B(material, pattern_256):
    """Provides a second, distinct shape with a 256x256 pattern."""
    return Shape(
        transform=jp.eye(4).at[0, 3].set(1.0),
        type=paz.graphics.CUBE,
        material=material,
        pattern=pattern_256,
    )


@pytest.fixture
def shape_512(material, pattern_512):
    """Provides a shape with a 512x512 pattern."""
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.PLANE,
        material=material,
        pattern=pattern_512,
    )


def test_compute_bounces_default(shape_256_A):
    """Tests that compute_bounces returns 1 for default materials."""
    shapes = [shape_256_A]
    assert paz.graphics.scene.compute_bounces(shapes) == 1


def test_compute_bounces_reflective():
    """Tests that compute_bounces returns 5 for reflective materials."""
    material = Material(reflective=0.5)
    shape = Shape(jp.eye(4), paz.graphics.SPHERE, material)
    assert paz.graphics.scene.compute_bounces([shape]) == 5


def test_compute_bounces_transparent():
    """Tests that compute_bounces returns 5 for transparent materials."""
    material = Material(transparency=0.5)
    shape = Shape(jp.eye(4), paz.graphics.SPHERE, material)
    assert paz.graphics.scene.compute_bounces([shape]) == 5


def test_compute_bounces_mixed(shape_256_A):
    """Returns 5 if any material is reflective or transparent."""
    material_ref = Material(reflective=0.5)
    shape_ref = Shape(jp.eye(4), paz.graphics.SPHERE, material_ref)
    shapes = [shape_256_A, shape_ref]
    assert paz.graphics.scene.compute_bounces(shapes) == 5


def test_compile_sorting(shape_256_A, shape_512, shape_256_B):
    """Shapes are grouped by pattern size and the masks follow them."""
    scene = paz.graphics.Scene([shape_256_A, shape_512, shape_256_B])
    lights = [paz.graphics.PointLight(jp.ones(3), jp.zeros(3))]
    mask = jp.array([True, False, True])
    shadow_mask = jp.array([False, True, False])
    compiled = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)
    shapes = compiled.shapes
    assert len(shapes) == 3
    assert shapes[0].type == paz.graphics.SPHERE
    assert shapes[1].type == paz.graphics.CUBE
    assert shapes[2].type == paz.graphics.PLANE
    assert jp.array_equal(compiled.mask, jp.array([True, True, False]))
    expected_shadow_mask = jp.array([False, False, True])
    assert jp.array_equal(compiled.shadow_mask, expected_shadow_mask)


def test_compile_rejects_mixed_shape_materials():
    phong_sphere = paz.graphics.Sphere(jp.eye(4), Material())
    cook_torrance = paz.graphics.CookTorranceMaterial()
    cook_torrance_sphere = paz.graphics.Sphere(jp.eye(4), cook_torrance)
    scene = paz.graphics.Scene([phong_sphere, cook_torrance_sphere])
    lights = [paz.graphics.PointLight(jp.ones(3), jp.zeros(3))]
    with pytest.raises(ValueError, match="must all be of one type"):
        paz.graphics.scene.compile(scene, lights, None)
