from collections import namedtuple

PointLight = namedtuple("PointLight", ["intensity", "position"])
material_properties = ["color", "ambient", "diffuse", "specular", "shininess"]
Material = namedtuple("Material", material_properties)
Pattern = namedtuple("Pattern", ["transform", "type", "image"])
Shape = namedtuple("Shape", ["transform", "type", "pattern", "material"])
