from paz.graphics.patterns.spherical import spherical_map
from paz.graphics.patterns.checker import compute_checker_colors


def compute_colors(points3D, color_A, color_B):
    u, v = spherical_map(points3D)
    colors = compute_checker_colors(u, v, color_A, color_B, 16, 16)
    return colors
