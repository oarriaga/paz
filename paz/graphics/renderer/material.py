from collections import namedtuple

import jax.numpy as jp

MATERIAL_NAMES = "reflectivities transparencies refractive_indices"
HitMaterial = namedtuple("HitMaterial", MATERIAL_NAMES.split())


def compute_material_properties(compiled, closest, triangle_hit):
    shape_material = gather_shape_material(compiled, closest.primitive_index)
    if triangle_hit is None:
        resolved = shape_material
    else:
        materials = compiled.triangles.materials
        triangle = gather_primitive_material(materials, triangle_hit.primitive)
        is_triangle = closest.primitive_index == len(compiled.shapes)
        resolved = select_material(is_triangle, triangle, shape_material)
    return resolved


def gather_shape_material(compiled, hit_shape_args):
    values = collect_shape_values(compiled)
    gathered = tuple(jp.array(row)[hit_shape_args] for row in values)
    return HitMaterial(*gathered)


def collect_shape_values(compiled):
    reflectivities, transparencies, refractive_indices = [], [], []
    for shape in compiled.shapes:
        reflectivities.append(shape.material.reflective)
        transparencies.append(shape.material.transparency)
        refractive_indices.append(shape.material.refractive_index)
    if compiled.triangles is not None:
        reflectivities.append(0.0)
        transparencies.append(0.0)
        refractive_indices.append(1.0)
    return reflectivities, transparencies, refractive_indices


def gather_primitive_material(materials, primitive):
    reflectivities = materials.reflective[primitive]
    transparencies = materials.transparency[primitive]
    refractive_indices = materials.refractive_index[primitive]
    return HitMaterial(reflectivities, transparencies, refractive_indices)


def select_material(is_triangle, triangle, shape):
    fields = []
    for triangle_field, shape_field in zip(triangle, shape):
        fields.append(jp.where(is_triangle, triangle_field, shape_field))
    return HitMaterial(*fields)
