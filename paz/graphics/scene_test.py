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


@pytest.fixture
def shape_default(material):
    """Provides a shape that uses the default 1x1 pattern."""
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.CONE,
        material=material,
        pattern=Pattern(),
    )


def test_group_by_size_with_empty_list():
    """Tests that an empty list of shapes results in an empty dictionary."""
    shapes_list = []
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)
    assert result == {}


def test_group_by_size_with_homogenous_list(shape_256_A, shape_256_B):
    """Tests grouping a list where all shapes have the same image size."""
    shapes_list = [shape_256_A, shape_256_B]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 1
    assert (256, 256) in result

    result_list = result[(256, 256)]
    assert len(result_list) == 2

    # FIX: Use `is` to check for object identity, avoiding the ValueError.
    assert result_list[0] is shape_256_A
    assert result_list[1] is shape_256_B


def test_group_by_size_with_heterogeneous_list(shape_256_A, shape_512):
    """Tests grouping a list with two different image sizes."""
    shapes_list = [shape_256_A, shape_512]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 2
    assert (256, 256) in result
    assert (512, 512) in result
    assert result[(256, 256)] == [shape_256_A]
    assert result[(512, 512)] == [shape_512]


def test_group_by_size_with_default_pattern(shape_default):
    """shapes with default patterns are grouped correctly by size (1, 1)."""
    shapes_list = [shape_default]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 1
    assert (1, 1) in result
    assert result[(1, 1)] == [shape_default]


def test_group_by_size_with_complex_mixed_list(
    shape_256_A, shape_512, shape_default, shape_256_B
):
    """Tests a complex mix of shapes with different and shared image sizes."""
    shapes_list = [shape_256_A, shape_512, shape_default, shape_256_B]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 3
    assert (256, 256) in result
    assert (512, 512) in result
    assert (1, 1) in result

    # Check the contents of each group
    group_256 = result[(256, 256)]
    assert len(group_256) == 2
    assert len(result[(512, 512)]) == 1
    assert len(result[(1, 1)]) == 1

    # FIX: Use `is` to check for object identity. The order is preserved.
    assert group_256[0] is shape_256_A
    assert group_256[1] is shape_256_B
    assert result[(512, 512)][0] is shape_512
    assert result[(1, 1)][0] is shape_default


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
    """Tests that compute_bounces returns 5 if any material is reflective or transparent."""
    material_ref = Material(reflective=0.5)
    shape_ref = Shape(jp.eye(4), paz.graphics.SPHERE, material_ref)
    shapes = [shape_256_A, shape_ref]
    assert paz.graphics.scene.compute_bounces(shapes) == 5