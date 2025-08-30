from collections import namedtuple
import jax.numpy as jp
from paz.graphics.constants import (
    SPHERE,
    CUBE,
    PLANE,
    CONE,
    CYLINDER,
    NO_PATTERN,
    DEFAULT_COLOR,
)

PointLight = namedtuple("PointLight", ["intensity", "position"])


Pattern = namedtuple(
    "Pattern",
    ["transform", "type", "image"],
    defaults=(jp.eye(4), NO_PATTERN, jp.ones((1, 1, 3))),
)


Material = namedtuple(
    "Material",
    ["color", "ambient", "diffuse", "specular", "shininess"],
    defaults=(DEFAULT_COLOR, 0.1, 0.9, 0.9, 200),
)

Shape = namedtuple(
    "Shape", ["transform", "type", "material", "pattern"], defaults=(Pattern(),)
)


Scene = namedtuple("Scene", ["nodes", "parent_array"])


Group = namedtuple("Group", ["shapes", "transform"])


def Sphere(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, SPHERE, material, pattern)


def Plane(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, PLANE, material, pattern)


def Cube(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, CUBE, material, pattern)


def Cone(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, CONE, material, pattern)


def Cylinder(transform=jp.eye(4), material=Material(), pattern=Pattern()):
    return Shape(transform, CYLINDER, material, pattern)
