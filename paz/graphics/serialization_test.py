import pytest
import jax
import jax.numpy as jp
import json
import paz
import numpy as np

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
    return types.Material(
        color=jp.array([1.0, 0.5, 0.0]),
        ambient=0.2,
        reflective=0.5,
        refractive=0.8,
        refractive_index=1.33,
    )


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


@pytest.fixture
def pattern_with_image():
    """Provides a Pattern with actual image data."""
    image = jp.ones((64, 64, 3))
    return types.Pattern(image=image, type=constants.PLANAR_PATTERN)


@pytest.fixture
def pattern_no_image():
    """Provides a Pattern that doesn't need image saving."""
    return types.Pattern(type=constants.NO_PATTERN)


@pytest.fixture
def shape_with_image(sample_material, pattern_with_image):
    """Provides a Shape using the pattern_with_image."""
    return types.Sphere(
        transform=jp.eye(4),
        material=sample_material,
        pattern=pattern_with_image,
    )


@pytest.fixture
def shape_no_image(sample_material, pattern_no_image):
    """Provides a Shape using the pattern_no_image."""
    return types.Cube(
        transform=jp.eye(4), material=sample_material, pattern=pattern_no_image
    )


@pytest.fixture
def sample_group_with_images(shape_with_image, shape_no_image):
    """
    A Group containing both a shape with an image pattern and one without.
    Useful for testing multiple assets in one save.
    """
    return types.Group(
        shapes=[shape_with_image, shape_no_image], transform=jp.eye(4)
    )


@pytest.fixture
def sample_scene_with_images(shape_with_image, sample_group_with_images):
    """
    A Scene containing a shape with an image pattern and a group that also has
    images. Tests nested asset saving.
    """
    nodes = [shape_with_image, sample_group_with_images]
    parent_array = jp.array([-1, 0])  # Example parent array
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
    reconstructed_shape = serialization.build_node(shape_data)[0]
    reconstructed_group = serialization.build_node(group_data)[0]
    assert isinstance(reconstructed_shape, types.Shape)
    assert isinstance(reconstructed_group, types.Group)


def test_full_scene_save_and_load_round_trip(tmp_path, sample_scene):
    """Tests a round-trip for a full Scene object."""
    filepath = tmp_path / "scene"
    serialization.save(filepath, sample_scene)
    loaded_scene = serialization.load(filepath)
    assert isinstance(loaded_scene, types.Scene)
    assert_pytrees_allclose(loaded_scene, sample_scene)


def test_save_and_load_standalone_group(tmp_path, sample_group):
    """Tests a round-trip for a standalone Group object."""
    filepath = tmp_path / "group"
    serialization.save(filepath, sample_group)
    loaded_group = serialization.load(filepath)
    assert isinstance(loaded_group, types.Group)
    assert_pytrees_allclose(loaded_group, sample_group)


def test_save_and_load_standalone_shape(tmp_path, sample_shape):
    """Tests a round-trip for a standalone Shape object."""
    filepath = tmp_path / "shape"
    serialization.save(filepath, sample_shape)
    loaded_shape = serialization.load(filepath)
    assert isinstance(loaded_shape, types.Shape)
    assert_pytrees_allclose(loaded_shape, sample_shape)


def test_load_raises_error_for_unknown_top_level_type(tmp_path):
    """Tests that load fails gracefully for an unknown JSON structure."""
    dir_path = tmp_path / "invalid_dir"
    dir_path.mkdir()
    filepath = dir_path / "configuration.json"
    invalid_data = {"some_other_key": "some_value"}
    with open(filepath, "w") as f:
        json.dump(invalid_data, f)
    with pytest.raises(
        TypeError, match="Data is not a valid Scene, Group or Shape"
    ):
        serialization.load(dir_path)


def test_save_creates_asset_file_and_json_path(tmp_path, shape_with_image):
    """
    Tests saving a shape with an image:
    1. Creates the image file in the same directory as the JSON.
    2. JSON contains the correct filename (string) for the image pattern.
    """
    scene_dir = tmp_path / "test_image_shape"
    serialization.save(str(scene_dir), shape_with_image)

    # 1. Verify JSON file content
    json_filepath = scene_dir / "configuration.json"
    assert json_filepath.is_file()
    with open(json_filepath, "r") as f:
        data = json.load(f)

    pattern_data = data["pattern"]
    assert isinstance(
        pattern_data["image"], str
    )  # Should be a filename string, not a list/array

    # Expected filename based on `dirname_pattern_X.png` convention
    expected_filename = f"{scene_dir.name}_pattern_0.png"
    assert pattern_data["image"] == expected_filename

    # 2. Verify image file exists in the same directory
    image_filepath = scene_dir / expected_filename
    assert image_filepath.is_file(), f"Image file not found at {image_filepath}"

    # Optional: Load and verify the image content (requires paz.image.load to work)
    loaded_image = paz.image.load(str(image_filepath))
    # Denormalize the original fixture image to compare with saved uint8
    original_image_denormalized = paz.image.denormalize(
        shape_with_image.pattern.image
    )
    assert np.allclose(loaded_image, original_image_denormalized)


def test_save_handles_multiple_image_patterns(
    tmp_path, shape_with_image, pattern_with_image
):
    """
    Tests that multiple shapes/patterns with image data within a single
    save operation result in uniquely named image files.
    """
    # Create a second shape with a unique image (or just reuse the fixture's image)
    # The counter in `serialization.py` should handle uniqueness.
    shape2 = types.Cube(
        transform=jp.eye(4),
        material=types.Material(color=jp.array([0.0, 1.0, 0.0])),
        pattern=pattern_with_image,  # Reusing the image data, but it'll be saved separately
    )

    scene = types.Scene(
        nodes=[shape_with_image, shape2], parent_array=jp.array([-1, 0])
    )

    scene_dir = tmp_path / "multi_pattern_scene"
    serialization.save(str(scene_dir), scene)

    json_filepath = scene_dir / "configuration.json"
    assert json_filepath.is_file()
    with open(json_filepath, "r") as f:
        data = json.load(f)

    # Verify first pattern's image filename
    pattern1_data = data["nodes"][0]["pattern"]
    expected_filename1 = f"{scene_dir.name}_pattern_0.png"
    assert pattern1_data["image"] == expected_filename1
    assert (scene_dir / expected_filename1).is_file()

    # Verify second pattern's image filename
    pattern2_data = data["nodes"][1]["pattern"]
    expected_filename2 = f"{scene_dir.name}_pattern_1.png"
    assert pattern2_data["image"] == expected_filename2
    assert (scene_dir / expected_filename2).is_file()


def test_save_handles_no_image_pattern(tmp_path, shape_no_image):
    """
    Tests that a Pattern without image data (e.g., NO_PATTERN type)
    does NOT create an image file, and its 'image' field remains a list (default value).
    """
    scene_dir = tmp_path / "no_image_pattern_shape"
    serialization.save(str(scene_dir), shape_no_image)

    json_filepath = scene_dir / "configuration.json"
    assert json_filepath.is_file()
    with open(json_filepath, "r") as f:
        data = json.load(f)

    pattern_data = data["pattern"]
    # For NO_PATTERN, the default image is usually a 1x1 white image array
    # which gets converted to a list by `jp.ndarray.tolist()`.
    assert isinstance(pattern_data["image"], list)
    assert len(pattern_data["image"]) > 0  # Should not be an empty list

    # Crucially, verify no image file was created for this pattern
    assert not any(
        f.name.startswith(f"{scene_dir.name}_pattern_")
        for f in scene_dir.iterdir()
    )


def test_full_scene_save_and_load_round_trip_with_images(
    tmp_path, sample_scene_with_images
):
    """
    Comprehensive round-trip test for a Scene containing nested shapes and groups,
    including those with image patterns, verifying full reconstruction.
    """
    scene_dir = tmp_path / "full_scene_test"
    serialization.save(str(scene_dir), sample_scene_with_images)
    loaded_scene = serialization.load(str(scene_dir))

    assert isinstance(loaded_scene, types.Scene)
    # assert_pytrees_allclose will recursively compare all fields, including loaded images
    assert_pytrees_allclose(loaded_scene, sample_scene_with_images)


def test_load_raises_error_if_asset_missing(tmp_path, shape_with_image):
    """
    Tests that `load` raises a FileNotFoundError if an expected image asset
    file (referenced in the JSON) is missing from the directory.
    """
    scene_dir = tmp_path / "missing_asset_test"
    serialization.save(str(scene_dir), shape_with_image)

    # Get the expected image filename from the saved JSON
    json_filepath = scene_dir / "configuration.json"
    with open(json_filepath, "r") as f:
        data = json.load(f)
    image_filename = data["pattern"]["image"]

    # Delete the image asset file
    image_filepath = scene_dir / image_filename
    assert image_filepath.is_file()  # Ensure it was created
    image_filepath.unlink()

    with pytest.raises(FileNotFoundError):
        serialization.load(str(scene_dir))


def test_material_save_and_load_round_trip(tmp_path, sample_material):
    """
    Tests a round-trip for a standalone Material object, ensuring all properties,
    including new ones, are correctly serialized and deserialized.
    """
    filepath = tmp_path / "material_test"
    # To save a Material, it needs to be part of a Shape or a similar structure
    # that `serialization.save` knows how to handle.
    # We can wrap it in a dummy Shape for this test.
    dummy_shape = types.Sphere(material=sample_material)
    serialization.save(filepath, dummy_shape)

    loaded_shape = serialization.load(filepath)
    assert isinstance(loaded_shape, types.Shape)
    assert_pytrees_allclose(loaded_shape.material, sample_material)