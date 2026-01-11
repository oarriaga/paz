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


@pytest.fixture
def valid_nodes(sample_material):
    """Provides a valid list of nodes for Scene creation."""
    shape_node = types.Sphere(material=sample_material)
    group_node = types.Group(shapes=[types.Cube()], transform=jp.eye(4))
    return [shape_node, group_node]


@pytest.fixture
def valid_parent_array():
    """Provides a valid parent array for a 2-node scene."""
    return jp.array([-1, 0])


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


def test_scene_creation_success(valid_nodes, valid_parent_array):
    """Tests that a Scene can be created with valid inputs."""
    scene = types.Scene(nodes=valid_nodes, parent_array=valid_parent_array)
    assert isinstance(scene, types.Scene)
    assert len(scene.nodes) == 2
    assert jp.allclose(scene.parent_array, valid_parent_array)


def test_scene_raises_error_for_non_list_nodes(valid_parent_array):
    """Tests that `nodes` must be a list."""
    with pytest.raises(TypeError, match="`nodes` must be a list"):
        types.Scene(nodes="not_a_list", parent_array=valid_parent_array)


def test_scene_raises_error_for_invalid_node_content(valid_parent_array):
    """Tests that all elements in `nodes` must be a Shape or Group."""
    with pytest.raises(TypeError, match="All elements in `nodes` must be"):
        types.Scene(
            nodes=[types.Sphere(), 123], parent_array=valid_parent_array
        )


def test_scene_raises_error_for_mismatched_lengths(valid_nodes):
    """Tests that `nodes` and `parent_array` must have the same length."""
    with pytest.raises(
        ValueError, match="Length of `nodes` and `parent_array`"
    ):
        types.Scene(nodes=valid_nodes, parent_array=jp.array([-1]))


@pytest.mark.skip(reason="Validation disabled for JAX JIT compatibility")
def test_scene_raises_error_for_invalid_parent_index(valid_nodes):
    """Tests that all parent indices must be valid."""
    # Index 5 is out of bounds for a 2-node scene
    invalid_parent_array = jp.array([-1, 5])
    with pytest.raises(ValueError, match="contains invalid indices"):
        types.Scene(nodes=valid_nodes, parent_array=invalid_parent_array)


def test_scene_is_immutable(valid_nodes, valid_parent_array):
    """Tests that Scene attributes cannot be changed after creation."""
    scene = types.Scene(nodes=valid_nodes, parent_array=valid_parent_array)
    with pytest.raises(AttributeError):
        scene.nodes = []
