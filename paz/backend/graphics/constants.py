import jax.numpy as jp

FARAWAY = 1000.0
EPSILON = 1e-3

RED = jp.array([1.0, 0.0, 0.0])
GREEN = jp.array([0.0, 1.0, 0.0])
BLUE = jp.array([0.0, 0.0, 1.0])
BLACK = jp.array([0.0, 0.0, 0.0])
WHITE = jp.array([1.0, 1.0, 1.0])
GRAY = jp.array([0.5, 0.5, 0.5])
PINK = jp.array([1.0, 0.078, 0.57])

NO_PATTERN = 0
SPHERICAL_PATTERN = 1
PLANAR_PATTERN = 2

SPHERE = 0
CUBE = 1
CYLINDER = 2
CONE = 3
PLANE = 4
