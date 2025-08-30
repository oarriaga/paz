import pytest
import jax
import jax.numpy as jp

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
    sphere = types.Sphere()
    cube = types.Cube()
    assert isinstance(sphere, types.Shape)
    assert sphere.type == constants.SPHERE
    assert cube.type == constants.CUBE


def test_to_json_handles_nested_scene(sample_scene):
    serialized = serialization.to_json(sample_scene)
    assert isinstance(serialized, dict)
    assert "nodes" in serialized
    assert "parent_array" in serialized
    assert isinstance(serialized["nodes"], list)
    assert isinstance(serialized["nodes"][0]["transform"], list)


def test_build_node_differentiates_shape_and_group(sample_scene):
    scene_as_dict = serialization.to_json(sample_scene)
    shape_data = scene_as_dict["nodes"][0]
    group_data = scene_as_dict["nodes"][1]
    reconstructed_shape = serialization.build_node(shape_data)
    reconstructed_group = serialization.build_node(group_data)
    assert isinstance(reconstructed_shape, types.Shape)
    assert isinstance(reconstructed_group, types.Group)


def test_full_scene_save_and_load_round_trip(tmp_path, sample_scene):
    filepath = tmp_path / "scene.json"
    serialization.save(filepath, sample_scene)
    loaded_scene = serialization.load(filepath)
    assert isinstance(loaded_scene, types.Scene)
    assert_pytrees_allclose(loaded_scene, sample_scene)
