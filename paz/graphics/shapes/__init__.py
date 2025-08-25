import jax
import jax.numpy as jp

from .sphere import intersect_canonical_sphere
from .cube import intersect_canonical_cube
from .cylinder import intersect_canonical_cylinder
from .cone import intersect_canonical_cone
from .plane import intersect_canonical_plane

from .sphere import compute_canonical_normals_sphere
from .cube import compute_canonical_normals_cube
from .cylinder import compute_canonical_normals_cylinder
from .cone import compute_canonical_normals_cone
from .plane import compute_canonical_normals_plane


intersection_cases = [
    intersect_canonical_sphere,
    intersect_canonical_cube,
    intersect_canonical_cylinder,
    intersect_canonical_cone,
    intersect_canonical_plane,
]

normal_cases = [
    compute_canonical_normals_sphere,
    compute_canonical_normals_cube,
    compute_canonical_normals_cylinder,
    compute_canonical_normals_cone,
    compute_canonical_normals_plane,
]


def _merge(*leafs):
    concatenated_leafs = []
    for leaf in leafs:
        concatenated_leafs.append(leaf)
    return jp.array(concatenated_leafs)


def merge(*shapes):
    return jax.tree.map(_merge, *shapes)


def _expand_leafs(leaf):
    return jp.expand_dims(jp.array(leaf), 0)


def expand(shape):
    return jax.tree.map(_expand_leafs, shape)
