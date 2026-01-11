from collections import namedtuple
import jax.numpy as jp
import paz
from paz.graphics.constants import (
    SPHERE,
    CUBE,
    PLANE,
    CONE,
    CYLINDER,
    NO_PATTERN,
    DEFAULT_COLOR,
    SPHERICAL_PATTERN,
    PLANAR_PATTERN,
    CYLINDRICAL_PATTERN,
)

PointLight = namedtuple("PointLight", ["intensity", "position"])


Pattern = namedtuple(
    "Pattern",
    ["transform", "type", "image"],
    defaults=(jp.eye(4), NO_PATTERN, jp.ones((1, 1, 3))),
)


def SphericalPattern(image, transform=jp.eye(4)):
    return Pattern(transform, SPHERICAL_PATTERN, image)


def PlanarPattern(image, transform=jp.eye(4)):
    return Pattern(transform, PLANAR_PATTERN, image)


def CylindricalPattern(image, transform=jp.eye(4)):
    return Pattern(transform, CYLINDRICAL_PATTERN, image)


Material = namedtuple(
    "Material",
    [
        "color",
        "ambient",
        "diffuse",
        "specular",
        "shininess",
        "reflective",
        "transparency",
        "refractive_index",
    ],
    defaults=(DEFAULT_COLOR, 0.1, 0.9, 0.9, 200.0, 0.0, 0.0, 1.0),
)

Shape = namedtuple(
    "Shape", ["transform", "type", "material", "pattern"], defaults=(Pattern(),)
)


_SceneBase = namedtuple("Scene", ["nodes", "parent_array"])


class Scene(_SceneBase):
    """A Scene graph containing nodes and their parent-child relationships."""

    def __new__(cls, nodes, parent_array=None):
        if not isinstance(nodes, list):
            raise TypeError("`nodes` must be a list of Shape or Group objects.")
        if not all(isinstance(node, (Shape, Group)) for node in nodes):
            raise TypeError("All elements in `nodes` must be a Shape or Group.")

        if parent_array is None:
            parent_array = jp.full(len(nodes), -1)
        else:
            parent_array = jp.array(parent_array)
            if parent_array.ndim != 1:
                raise ValueError("`parent_array` must be a 1D array.")

            if len(nodes) != len(parent_array):
                raise ValueError(
                    "Length of `nodes` and `parent_array` must equal."
                )

            # is_root = parent_array == -1
            # is_valid_child = (parent_array >= 0) & (parent_array < len(nodes))
            # if not jp.all(is_root | is_valid_child):
            #     raise ValueError("`parent_array` contains invalid indices.")

        return super().__new__(cls, nodes, parent_array)


Group = namedtuple("Group", ["shapes", "transform"], defaults=(jp.eye(4),))

CSG = namedtuple("CSG", ["shape_A", "shape_B", "operation"])


def Sphere(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, SPHERE, material, pattern)


def Plane(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, PLANE, material, pattern)


def Cube(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, CUBE, material, pattern)


def Cone(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    canonical_scale = paz.SE3.scaling(jp.array([1.0, 2.0, 1.0]))
    canonical_shift = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))
    canonical_transform = canonical_shift @ canonical_scale
    return Shape(transform @ canonical_transform, CONE, material, pattern)


def Cylinder(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, CYLINDER, material, pattern)
