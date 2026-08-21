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