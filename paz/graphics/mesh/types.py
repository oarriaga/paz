from collections import namedtuple

import jax.numpy as jp

from paz.graphics.constants import NO_PATTERN
import paz


Mesh = namedtuple(
    "Mesh",
    [
        "vertices",
        "vertex_colors",
        "transform",
        "material",
        "faces",
        "edges",
        "pattern",
        "vertex_uvs",
    ],
    defaults=(None, None),
)


# TODO split this into two functions one for each branch.
def normalize_mesh_batch(meshes):
    num_meshes = meshes.vertices.shape[0]
    vertex_uvs = meshes.vertex_uvs
    if vertex_uvs is None:
        num_vertices = meshes.vertices.shape[1]
        vertex_uvs = jp.zeros((num_meshes, num_vertices, 2))

    pattern = meshes.pattern
    if pattern is None:
        pattern_transform = jp.repeat(
            jp.expand_dims(jp.eye(4), axis=0),
            repeats=num_meshes,
            axis=0,
        )
        pattern_type = jp.full((num_meshes,), NO_PATTERN)
        pattern_image = jp.ones((num_meshes, 1, 1, 3))
        pattern = paz.graphics.Pattern(
            pattern_transform, pattern_type, pattern_image
        )
    else:
        pattern_transform = pattern.transform
        if pattern_transform.ndim == 2:
            pattern_transform = jp.repeat(
                jp.expand_dims(pattern_transform, axis=0),
                repeats=num_meshes,
                axis=0,
            )
        pattern_type = pattern.type
        if jp.ndim(pattern_type) == 0:
            pattern_type = jp.full((num_meshes,), pattern_type)
        pattern_image = pattern.image
        if pattern_image.ndim == 3:
            pattern_image = jp.repeat(
                jp.expand_dims(pattern_image, axis=0),
                repeats=num_meshes,
                axis=0,
            )
        pattern = paz.graphics.Pattern(
            pattern_transform, pattern_type, pattern_image
        )
    return meshes._replace(pattern=pattern, vertex_uvs=vertex_uvs)
