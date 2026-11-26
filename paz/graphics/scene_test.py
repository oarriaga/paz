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
    """Tests that compile sorts shapes and masks correctly."""
    # Create scene with interleaved shapes: 256_A (0), 512 (1), 256_B (2)
    scene = paz.graphics.Scene([shape_256_A, shape_512, shape_256_B])
    lights = [paz.graphics.PointLight(jp.ones(3), jp.zeros(3))]
    
    # Masks reflect the indices
    mask = jp.array([True, False, True]) 
    shadow_mask = jp.array([False, True, False]) 

    # Compile
    compiled = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)
    shapes = compiled.shapes
    mask_out = compiled.mask
    shadow_mask_out = compiled.shadow_mask

    # Expected grouping:
    # (256, 256) group appears first because shape_256_A is first.
    # Group should contain [shape_256_A, shape_256_B]
    # (512, 512) group appears second.
    # Group should contain [shape_512]
    
    assert len(shapes) == 3
    # Check shapes order
    assert shapes[0].pattern.image.shape == (256, 256, 3)
    assert shapes[1].pattern.image.shape == (256, 256, 3)
    assert shapes[2].pattern.image.shape == (512, 512, 3)

    # Check masks were reordered correctly
    # shapes[0] is 256_A (original idx 0) -> mask True
    # shapes[1] is 256_B (original idx 2) -> mask True
    # shapes[2] is 512   (original idx 1) -> mask False
    assert mask_out[0] == True
    assert mask_out[1] == True
    assert mask_out[2] == False
    
    # Check shadow masks were reordered correctly
    # shapes[0] is 256_A (original idx 0) -> shadow_mask False
    # shapes[1] is 256_B (original idx 2) -> shadow_mask False
    # shapes[2] is 512   (original idx 1) -> shadow_mask True
    assert shadow_mask_out[0] == False
    assert shadow_mask_out[1] == False
    assert shadow_mask_out[2] == True


def test_nested_mask_expands_per_leaf(shape_256_A, shape_256_B):
    group = paz.graphics.Group([shape_256_A, shape_256_B])
    csg = paz.graphics.CSG(shape_256_A, group, paz.graphics.DIFFERENCE)
    scene = paz.graphics.Scene([shape_256_B, csg, group])
    mask = [True, [False, [True, False]], False]
    expanded = paz.graphics.scene.expand_mask(mask, scene)
    expected = jp.array([True, False, True, False, False, False])
    assert jp.array_equal(expanded, expected)


def test_nested_mask_length_must_match_children(shape_256_A, shape_256_B):
    csg = paz.graphics.CSG(shape_256_A, shape_256_B, paz.graphics.UNION)
    scene = paz.graphics.Scene([csg])
    with pytest.raises(ValueError):
        paz.graphics.scene.expand_mask([[True, False, True]], scene)


def test_csg_leaves_follow_sorted_shapes(shape_256_A, shape_512, shape_256_B):
    group = paz.graphics.Group([shape_512, shape_256_B])
    csg = paz.graphics.CSG(shape_256_A, group, paz.graphics.DIFFERENCE)
    scene = paz.graphics.Scene([shape_512, csg])
    lights = [paz.graphics.PointLight(jp.ones(3), jp.zeros(3))]
    compiled = paz.graphics.scene.compile(scene, lights, None)
    # Sorting by pattern size puts both 512 shapes before the 256 ones.
    union = paz.graphics.CSG(1, 3, paz.graphics.UNION)
    expected = paz.graphics.CSG(2, union, paz.graphics.DIFFERENCE)
    assert compiled.csg_trees == [expected]
    assert compiled.shapes[2].type == shape_256_A.type
    assert compiled.shapes[3].type == shape_256_B.type


def test_compile_plain_scene_has_no_csg_trees(shape_256_A, shape_256_B):
    scene = paz.graphics.Scene([shape_256_A, shape_256_B])
    lights = [paz.graphics.PointLight(jp.ones(3), jp.zeros(3))]
    assert paz.graphics.scene.compile(scene, lights, None).csg_trees == []


def test_empty_group_keeps_masks_aligned(shape_256_A, shape_256_B):
    nodes = [paz.graphics.Group([]), shape_256_A, shape_256_B]
    mask = [True, False, True]
    expanded = paz.graphics.scene.expand_mask(mask, paz.graphics.Scene(nodes))
    assert jp.array_equal(expanded, jp.array([False, True]))


def test_nested_mask_accepts_arrays(shape_256_A, shape_256_B):
    csg = paz.graphics.CSG(shape_256_A, shape_256_B, paz.graphics.UNION)
    scene = paz.graphics.Scene([csg, shape_256_A])
    mask = [jp.array([True, False]), True]
    expanded = paz.graphics.scene.expand_mask(mask, scene)
    assert jp.array_equal(expanded, jp.array([True, False, True]))
