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


_Pattern = namedtuple(
    "Pattern",
    ["transform", "type", "image"],
    defaults=(None, NO_PATTERN, None),
)


class Pattern(_Pattern):
    __slots__ = ()

    def __new__(cls, transform=None, type=NO_PATTERN, image=None):
        transform = jp.eye(4) if transform is None else jp.array(transform)
        image = jp.ones((1, 1, 3)) if image is None else jp.array(image)
        return super().__new__(cls, transform, type, image)


def SphericalPattern(image, transform=None):
    if transform is None:
        transform = jp.eye(4)
    return Pattern(transform, SPHERICAL_PATTERN, image)


def PlanarPattern(image, transform=None):
    if transform is None:
        transform = jp.eye(4)
    return Pattern(transform, PLANAR_PATTERN, image)


def CylindricalPattern(image, transform=None):
    if transform is None:
        transform = jp.eye(4)
    return Pattern(transform, CYLINDRICAL_PATTERN, image)


_Material = namedtuple(
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
    defaults=(None, 0.1, 0.9, 0.9, 200.0, 0.0, 0.0, 1.0),
)


class Material(_Material):
    __slots__ = ()

    def __new__(
        cls,
        color=None,
        ambient=0.1,
        diffuse=0.9,
        specular=0.9,
        shininess=200.0,
        reflective=0.0,
        transparency=0.0,
        refractive_index=1.0,
    ):
        color = jp.array(DEFAULT_COLOR) if color is None else jp.array(color)
        return super().__new__(
            cls,
            color,
            ambient,
            diffuse,
            specular,
            shininess,
            reflective,
            transparency,
            refractive_index,
        )


_Shape = namedtuple(
    "Shape",
    ["transform", "type", "material", "pattern"],
    defaults=(None,),
)


class Shape(_Shape):
    __slots__ = ()

    def __new__(cls, transform, type, material, pattern=None):
        transform = jp.array(transform)
        pattern = Pattern() if pattern is None else pattern
        return super().__new__(cls, transform, type, material, pattern)


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

            is_root = parent_array == -1
            is_valid_child = (parent_array >= 0) & (
                parent_array < len(nodes)
            )
            if not jp.all(is_root | is_valid_child):
                raise ValueError("`parent_array` contains invalid indices.")

        return super().__new__(cls, nodes, parent_array)


_Group = namedtuple("Group", ["shapes", "transform"], defaults=(None,))


class Group(_Group):
    __slots__ = ()

    def __new__(cls, shapes, transform=None):
        transform = jp.eye(4) if transform is None else jp.array(transform)
        return super().__new__(cls, shapes, transform)

CSG = namedtuple("CSG", ["shape_A", "shape_B", "operation"])


def Sphere(transform=None, material=Material(), pattern=Pattern()):
    if transform is None:
        transform = jp.eye(4)
    return Shape(transform, SPHERE, material, pattern)


def Plane(transform=None, material=Material(), pattern=Pattern()):
    if transform is None:
        transform = jp.eye(4)
    return Shape(transform, PLANE, material, pattern)


def Cube(transform=None, material=Material(), pattern=Pattern()):
    if transform is None:
        transform = jp.eye(4)
    return Shape(transform, CUBE, material, pattern)


def Cone(transform=None, material=Material(), pattern=Pattern()):
    if transform is None:
        transform = jp.eye(4)
    canonical_scale = paz.SE3.scaling(jp.array([1.0, 2.0, 1.0]))
    canonical_shift = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))
    canonical_transform = canonical_shift @ canonical_scale
    return Shape(transform @ canonical_transform, CONE, material, pattern)


def Cylinder(transform=None, material=Material(), pattern=Pattern()):
    if transform is None:
        transform = jp.eye(4)
    return Shape(transform, CYLINDER, material, pattern)
