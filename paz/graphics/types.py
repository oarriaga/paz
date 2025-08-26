import jax.numpy as jp
from paz.graphics.constants import NO_PATTERN
from collections import namedtuple

PointLight = namedtuple("PointLight", ["intensity", "position"])
material_properties = ["color", "ambient", "diffuse", "specular", "shininess"]
Material = namedtuple("Material", material_properties)

Pattern = namedtuple(
    "Pattern",
    ["transform", "type", "image"],
    defaults=(jp.eye(4), NO_PATTERN, jp.ones((1, 1, 3))),
)

Shape = namedtuple(
    "Shape", ["transform", "type", "material", "pattern"], defaults=(Pattern(),)
)


Group = namedtuple("Group", ["shapes", "parent_array"])
