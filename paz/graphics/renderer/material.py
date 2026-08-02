from collections import namedtuple

import jax.numpy as jp

MATERIAL_NAMES = "reflectivities transparencies refractive_indices"
HitMaterial = namedtuple("HitMaterial", MATERIAL_NAMES.split())


def compute_material_properties(compiled, hit_shape_args):
    values = collect_material_values(compiled)
    gathered = tuple(jp.array(row)[hit_shape_args] for row in values)
    return HitMaterial(*gathered)


def collect_material_values(compiled):
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
