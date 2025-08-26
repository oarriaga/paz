# change API so that user does not need to call shapes.merge
# change API so that an empty pattern is created by default
#   - maybe have a function called EmptyPattern
# change API so that the default masks is always all ones
# can we initialize namedtuples to have default values?


import jax.numpy as jp
import paz
from paz.graphics import NO_PATTERN
from paz import SE3


def build_empty_pattern():
    return paz.graphics.Pattern(jp.eye(4), NO_PATTERN, jp.ones((1, 1, 3)))


def build_material(color):
    return paz.graphics.Material(color, 0.1, 0.1, 0.0, 200.0)


def build_shape(transform, shape, pattern, material):
    return paz.graphics.Shape(transform, shape, pattern, material)


left = -2.0
radius = 0.5
shape_pose = SE3.rotation_z(jp.pi / 4.0) @ SE3.translation(
    jp.array([left, 0.0, radius])
)


scene = {
    "shapes": {
        "sphere": build_shape(
            jp.eye(4),
            shape_pose @ SE3.scaling(jp.array([0.5, 1.0, 0.5])),
            paz.graphics.SPHERE,
            build_empty_pattern(),
            build_material(jp.array([1.0, 0.0, 0.0])),
        ),
        "cube": build_shape(
            SE3.rotation_x(jp.pi / 2.0),
            paz.graphics.CUBE,
            build_empty_pattern(),
            build_material(jp.array([0.0, 1.0, 0.0])),
        ),
    },
    "lights": [
        paz.graphics.PointLight(jp.full((3,), 3.0), jp.array([1.0, 1.0, 1.0]))
    ],
}
