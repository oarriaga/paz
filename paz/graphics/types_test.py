import pytest
import jax.numpy as jp

from paz.graphics import types
from paz.graphics import constants


@pytest.fixture
def sample_material():
    """Provides a sample Material object."""
    return types.Material(
        color=jp.array([1.0, 0.5, 0.0]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.5,
        shininess=200.0,
    )


@pytest.fixture
def sample_transform():
    """Provides a sample 4x4 transform matrix (translation)."""
    return jp.eye(4).at[0, 3].set(10.0)


@pytest.fixture
def custom_pattern(sample_transform):
    """Provides a non-default Pattern object."""
    image = jp.zeros((16, 16, 3))
    return types.Pattern(
        transform=sample_transform,
        type=constants.SPHERICAL_PATTERN,
        image=image,
    )


def test_point_light_creation():
    """Tests that a PointLight can be created with correct attributes."""
    light = types.PointLight(
        intensity=jp.array([1.0, 1.0, 1.0]),
        position=jp.array([0.0, 10.0, -10.0]),
    )
    assert jp.allclose(light.intensity, jp.array([1.0, 1.0, 1.0]))
    assert jp.allclose(light.position, jp.array([0.0, 10.0, -10.0]))


def test_material_creation(sample_material):
    """Tests that a Material can be created with correct attributes."""
    assert jp.allclose(sample_material.color, jp.array([1.0, 0.5, 0.0]))
    assert sample_material.diffuse == 0.9


def test_material_is_immutable(sample_material):
    """Tests that Material attributes cannot be changed after creation."""
    with pytest.raises(AttributeError):
        sample_material.color = jp.array([0.0, 0.0, 0.0])


def test_pattern_default_transform():
    """Tests that a Pattern gets a default identity matrix for its transform."""
    pattern = types.Pattern()
    assert jp.allclose(pattern.transform, jp.eye(4))


def test_pattern_default_type():
    """Tests that a Pattern gets a default type of NO_PATTERN."""
    pattern = types.Pattern()
    assert pattern.type == constants.NO_PATTERN


def test_pattern_default_image():
    """Tests that a Pattern gets a default 1x1 white image."""
    pattern = types.Pattern()
    assert jp.allclose(pattern.image, jp.ones((1, 1, 3)))


def test_pattern_instantiation_with_custom_transform(sample_transform):
    """Tests that a Pattern's transform can be set at instantiation."""
    pattern = types.Pattern(transform=sample_transform)
    assert jp.allclose(pattern.transform, sample_transform)
    assert pattern.type == constants.NO_PATTERN  # Other fields remain default


def test_shape_creation(sample_transform, sample_material, custom_pattern):
    """Tests that a Shape can be created with all attributes specified."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.SPHERE,
        material=sample_material,
        pattern=custom_pattern,
    )
    assert shape.type == constants.SPHERE
    assert shape.material == sample_material
    assert shape.pattern == custom_pattern


def test_shape_instantiates_with_default_pattern(
    sample_transform, sample_material
):
    """Tests that a Shape gets a default Pattern when none is provided."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.CUBE,
        material=sample_material,
    )
    assert isinstance(shape.pattern, types.Pattern)


def test_shape_default_pattern_has_correct_type(
    sample_transform, sample_material
):
    """Tests the type of the default pattern within a new Shape."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.CUBE,
        material=sample_material,
    )
    assert shape.pattern.type == constants.NO_PATTERN


def test_shape_default_pattern_has_correct_transform(
    sample_transform, sample_material
):
    """Tests the transform of the default pattern within a new Shape."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.CUBE,
        material=sample_material,
    )
    assert jp.allclose(shape.pattern.transform, jp.eye(4))


def test_shape_overrides_default_pattern(
    sample_transform, sample_material, custom_pattern
):
    """Tests that a provided pattern correctly overrides the default."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.SPHERE,
        material=sample_material,
        pattern=custom_pattern,
    )
    assert shape.pattern.type == constants.SPHERICAL_PATTERN
    assert not jp.allclose(shape.pattern.transform, jp.eye(4))


def test_shape_is_immutable(sample_transform, sample_material):
    """Tests that Shape attributes cannot be changed after creation."""
    shape = types.Shape(
        transform=sample_transform,
        type=constants.CUBE,
        material=sample_material,
    )
    with pytest.raises(AttributeError):
        shape.type = constants.SPHERE
