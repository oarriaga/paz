import pytest
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
def sample_light():
    """Provides a sample PointLight object."""
    return PointLight(
        intensity=jp.array([1.0, 1.0, 1.0]), position=jp.array([10.0, 5.0, 0.0])
    )


@pytest.fixture
def sample_material():
    """Provides a sample Material object."""
    return Material(
        color=jp.array([1.0, 0.0, 0.0]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.3,
        shininess=200.0,
    )


@pytest.fixture
def sample_pattern():
    """Provides a sample Pattern object."""
    return Pattern(
        transform=jp.eye(4).at[0, 3].set(5.0),
        type=NO_PATTERN,
        image=jp.ones((2, 2, 3)),
    )


@pytest.fixture
def sample_shape(sample_material, sample_pattern):
    """Provides a single sample Shape object."""
    return Shape(
        transform=jp.eye(4),
        type=SPHERE,
        material=sample_material,
        pattern=sample_pattern,
    )


@pytest.fixture
def sample_shape_list(sample_material, sample_pattern):
    """Provides a list of sample Shape objects."""
    shape1 = Shape(jp.eye(4), SPHERE, sample_material, sample_pattern)
    shape2 = Shape(
        jp.eye(4).at[1, 3].set(2.0), CUBE, sample_material, sample_pattern
    )
    return [shape1, shape2]


@pytest.fixture
def sample_group(sample_shape_list):
    """Provides a sample Group object."""
    return Group(shapes=sample_shape_list, parent_array=jp.array([-1, 0]))


@pytest.fixture
def sample_camera_pose():
    """Provides a sample 4x4 camera pose matrix."""
    return jp.eye(4).at[:3, 3].set(jp.array([0.0, 2.0, -5.0]))


def test_serialize_jax_array():
    """Tests that a JAX array is serialized to a list."""
    array = jp.array([[1.0, 2.0], [3.0, 4.0]])
    serialized = serialization.serialize(array)
    assert isinstance(serialized, list)
    assert serialized == [[1.0, 2.0], [3.0, 4.0]]


def test_serialize_namedtuple(sample_light):
    """Tests that a namedtuple is serialized to a dictionary."""
    serialized = serialization.serialize(sample_light)
    assert isinstance(serialized, dict)
    assert isinstance(serialized["position"], list)


def test_serialize_nested_namedtuple(sample_shape):
    """Tests that a nested namedtuple is recursively serialized to dicts."""
    serialized = serialization.serialize(sample_shape)
    assert isinstance(serialized, dict)
    assert isinstance(serialized["material"], dict)
    assert isinstance(serialized["material"]["color"], list)


def test_save_and_load_group(
    tmp_path, sample_group, sample_light, sample_camera_pose
):
    """Tests a full save/load round-trip for a scene with a Group."""
    filepath = tmp_path / "scene.json"
    serialization.save(
        filepath,
        shapes=sample_group,
        lights=[sample_light],
        camera_pose=sample_camera_pose,
    )
    loaded_data = serialization.load(filepath)

    assert "shapes" in loaded_data
    assert "lights" in loaded_data
    assert "camera_pose" in loaded_data

    assert_pytrees_allclose(loaded_data["shapes"], sample_group)
    assert_pytrees_allclose(loaded_data["lights"][0], sample_light)
    assert_pytrees_allclose(loaded_data["camera_pose"], sample_camera_pose)


def test_save_and_load_shape_list(tmp_path, sample_shape_list, sample_light):
    """Tests a full save/load round-trip for a scene with a list of Shapes."""
    filepath = tmp_path / "scene.json"
    serialization.save(
        filepath, shapes=sample_shape_list, lights=[sample_light]
    )
    loaded_data = serialization.load(filepath)

    assert isinstance(loaded_data["shapes"], list)
    assert len(loaded_data["shapes"]) == len(sample_shape_list)
    assert_pytrees_allclose(loaded_data["shapes"][0], sample_shape_list[0])
    assert_pytrees_allclose(loaded_data["shapes"][1], sample_shape_list[1])


def test_save_and_load_single_shape(tmp_path, sample_shape):
    """Tests a full save/load round-trip for a scene with a single Shape."""
    filepath = tmp_path / "scene.json"
    serialization.save(filepath, shapes=sample_shape)
    loaded_data = serialization.load(filepath)

    assert isinstance(loaded_data["shapes"], Shape)
    assert_pytrees_allclose(loaded_data["shapes"], sample_shape)


def test_load_raises_error_for_unknown_structure(tmp_path):
    """Tests that load fails gracefully with an unknown data structure."""
    filepath = tmp_path / "bad_scene.json"
    with open(filepath, "w") as f:
        f.write('{"shapes": 123}')  # 123 is not a valid shape/group/list

    with pytest.raises(TypeError):
        serialization.load(filepath)
