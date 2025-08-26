import pytest
import json
import jax
import jax.numpy as jp
from paz.graphics import serialization
from paz.graphics.types import PointLight, Material, Pattern, Shape, Group
from paz.graphics.constants import SPHERE, CUBE, NO_PATTERN


def assert_pytrees_allclose(a, b):
    a_leaves = jax.tree_util.tree_leaves(a)
    b_leaves = jax.tree_util.tree_leaves(b)
    assert len(a_leaves) == len(b_leaves)
    for leaf_a, leaf_b in zip(a_leaves, b_leaves):
        if isinstance(leaf_a, jp.ndarray):
            assert jp.allclose(leaf_a, leaf_b)
        else:
            assert leaf_a == leaf_b


@pytest.fixture
def sample_group():
    material = Material(jp.zeros(3), 0.1, 0.9, 0.3, 200.0)
    pattern = Pattern(jp.eye(4), NO_PATTERN, jp.ones((1, 1, 3)))
    shape1 = Shape(jp.eye(4), SPHERE, material, pattern)
    shape2 = Shape(jp.eye(4), CUBE, material, pattern)
    return Group(shapes=[shape1, shape2], parent_array=jp.array([-1, 0]))


@pytest.fixture
def sample_light():
    return PointLight(jp.ones(3), jp.ones(3) * 10)


@pytest.fixture
def sample_camera_pose():
    return jp.eye(4)


def test_save_raises_error_for_invalid_extension(tmp_path):
    """Tests that save() fails if the filepath is not .json."""
    filepath = tmp_path / "scene.txt"
    with pytest.raises(ValueError, match="must have a .json extension"):
        serialization.save(filepath, data="test")


def test_save_writes_valid_json_file(tmp_path, sample_light):
    """Tests that the output file is a well-formed JSON."""
    filepath = tmp_path / "test.json"
    serialization.save(filepath, light=sample_light)
    # The test passes if this line doesn't raise a json.JSONDecodeError
    with open(filepath, "r") as f:
        json.load(f)


def test_load_returns_single_object_for_single_key_file(tmp_path, sample_group):
    """Tests that load() returns a direct object when the JSON has one key."""
    filepath = tmp_path / "scene.json"
    serialization.save(filepath, shapes=sample_group)

    loaded_object = serialization.load(filepath)

    assert isinstance(loaded_object, Group)
    assert not isinstance(loaded_object, dict)
    assert_pytrees_allclose(loaded_object, sample_group)


def test_load_returns_dictionary_for_multiple_key_file(
    tmp_path, sample_group, sample_light, sample_camera_pose
):
    """Tests that load() returns a dict when the JSON has multiple keys."""
    filepath = tmp_path / "scene.json"
    serialization.save(
        filepath,
        shapes=sample_group,
        lights=[sample_light],
        camera_pose=sample_camera_pose,
    )
    loaded_data = serialization.load(filepath)

    assert isinstance(loaded_data, dict)
    assert "shapes" in loaded_data
    assert "lights" in loaded_data
    assert "camera_pose" in loaded_data
    assert_pytrees_allclose(loaded_data["shapes"], sample_group)
