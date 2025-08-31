import pytest
import jax
import jax.numpy as jp
import json

from paz.graphics import types
from paz.graphics import serialization
from paz.graphics import constants


def assert_pytrees_allclose(a, b):
    a_leaves, a_treedef = jax.tree_util.tree_flatten(a)
    b_leaves, b_treedef = jax.tree_util.tree_flatten(b)
    assert a_treedef == b_treedef
    for leaf_a, leaf_b in zip(a_leaves, b_leaves):
        if isinstance(leaf_a, jp.ndarray):
            assert jp.allclose(leaf_a, leaf_b)
        else:
            assert leaf_a == leaf_b


@pytest.fixture
def sample_material():
    return types.Material(color=jp.array([1.0, 0.5, 0.0]), ambient=0.2)


@pytest.fixture
def sample_shape(sample_material):
    return types.Sphere(transform=jp.eye(4), material=sample_material)


@pytest.fixture
def sample_group(sample_material):
    cube = types.Cube(transform=jp.eye(4).at[0, 3].set(1.0))
    plane = types.Plane(material=sample_material)
    return types.Group(shapes=[cube, plane], transform=jp.eye(4))


@pytest.fixture
def sample_scene(sample_shape, sample_group):
    nodes = [sample_shape, sample_group]
    parent_array = jp.array([-1, 0])
    return types.Scene(nodes=nodes, parent_array=parent_array)


def test_shape_factory_creates_correct_type():
    """Tests that the Shape constructor helpers set the correct type."""
    sphere = types.Sphere()
    cube = types.Cube()
    assert isinstance(sphere, types.Shape)
    assert sphere.type == constants.SPHERE
    assert cube.type == constants.CUBE


def test_is_group_identifier(sample_group, sample_shape):
    """Tests the is_group helper function."""
    group_dict = serialization.to_json(sample_group)
    shape_dict = serialization.to_json(sample_shape)
    assert serialization.is_group(group_dict)
    assert not serialization.is_group(shape_dict)


def test_is_shape_identifier(sample_group, sample_shape):
    """Tests the is_shape helper function."""
    group_dict = serialization.to_json(sample_group)
    shape_dict = serialization.to_json(sample_shape)
    assert not serialization.is_shape(group_dict)
    assert serialization.is_shape(shape_dict)


def test_is_scene_identifier(sample_scene, sample_group):
    """Tests the is_scene helper function."""
    scene_dict = serialization.to_json(sample_scene)
    group_dict = serialization.to_json(sample_group)
    assert serialization.is_scene(scene_dict)
    assert not serialization.is_scene(group_dict)


def test_to_json_handles_nested_scene(sample_scene):
    """Tests that the to_json helper correctly serializes a Scene."""
    serialized = serialization.to_json(sample_scene)
    assert isinstance(serialized, dict)
    assert "nodes" in serialized
    assert "parent_array" in serialized


def test_build_node_differentiates_shape_and_group(sample_scene):
    """Tests that build_node correctly identifies and constructs Shapes vs Groups."""
    scene_as_dict = serialization.to_json(sample_scene)
    shape_data = scene_as_dict["nodes"][0]
    group_data = scene_as_dict["nodes"][1]
    reconstructed_shape = serialization.build_node(shape_data)
    reconstructed_group = serialization.build_node(group_data)
    assert isinstance(reconstructed_shape, types.Shape)
    assert isinstance(reconstructed_group, types.Group)


def test_full_scene_save_and_load_round_trip(tmp_path, sample_scene):
    """Tests a round-trip for a full Scene object."""
    filepath = tmp_path / "scene.json"
    serialization.save(filepath, sample_scene)
    loaded_scene = serialization.load(filepath)
    assert isinstance(loaded_scene, types.Scene)
    assert_pytrees_allclose(loaded_scene, sample_scene)


def test_save_and_load_standalone_group(tmp_path, sample_group):
    """Tests a round-trip for a standalone Group object."""
    filepath = tmp_path / "group.json"
    serialization.save(filepath, sample_group)
    loaded_group = serialization.load(filepath)
    assert isinstance(loaded_group, types.Group)
    assert_pytrees_allclose(loaded_group, sample_group)


def test_save_and_load_standalone_shape(tmp_path, sample_shape):
    """Tests a round-trip for a standalone Shape object."""
    filepath = tmp_path / "shape.json"
    serialization.save(filepath, sample_shape)
    loaded_shape = serialization.load(filepath)
    assert isinstance(loaded_shape, types.Shape)
    assert_pytrees_allclose(loaded_shape, sample_shape)


def test_load_raises_error_for_unknown_top_level_type(tmp_path):
    """Tests that load fails gracefully for an unknown JSON structure."""
    filepath = tmp_path / "invalid.json"
    invalid_data = {"some_other_key": "some_value"}
    with open(filepath, "w") as f:
        json.dump(invalid_data, f)
    with pytest.raises(
        TypeError, match="Data is not a valid Scene, Group or Shape"
    ):
        serialization.load(filepath)
