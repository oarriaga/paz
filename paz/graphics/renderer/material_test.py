import jax.numpy as jp

import paz
from paz.graphics.renderer import material
from paz.graphics.types import Material, Sphere, Scene


def test_gather_shape_material():
    material_A = Material(reflective=0.5)
    material_B = Material(transparency=0.8)
    scene = Scene([Sphere(material=material_A), Sphere(material=material_B)])
    compiled = paz.graphics.scene.compile(scene, [], None)
    indices = jp.array([0, 1])
    gathered = material.gather_shape_material(compiled, indices)
    assert gathered.reflectivities[0] == 0.5
    assert gathered.transparencies[1] == 0.8
    assert gathered.refractive_indices[0] == 1.0


def test_gather_primitive_material_reads_mesh_fields():
    materials = Material(
        reflective=jp.array([0.0, 0.6]),
        transparency=jp.array([0.0, 0.3]),
        refractive_index=jp.array([1.0, 1.5]),
    )
    primitive = jp.array([1, 0, 1])
    gathered = material.gather_primitive_material(materials, primitive)
    expected_indices = jp.array([1.5, 1.0, 1.5])
    expected_reflectivities = jp.array([0.6, 0.0, 0.6])
    assert jp.array_equal(gathered.reflectivities, expected_reflectivities)
    assert jp.array_equal(gathered.refractive_indices, expected_indices)
