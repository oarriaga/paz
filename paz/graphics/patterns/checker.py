import jax.numpy as jp


def compute_colors(points3D, color_A, color_B):
    floor_points = jp.floor(points3D)
    floor_points = jp.sum(floor_points, axis=1)
    choose_color_A = jp.remainder(floor_points, 2)
    choose_color_A = choose_color_A.astype('bool')
    colors = jp.vstack([color_A, color_B])
    stripe_colors = jp.take(colors, choose_color_A, axis=0)
    return stripe_colors


def compute_checker_colors(u, v, color_A, color_B, H, W):
    u = jp.floor(u * W)
    v = jp.floor(v * H)
    choose_color_A = jp.remainder(u + v, 2)
    choose_color_A = choose_color_A.astype('bool')
    choose_color_A = jp.squeeze(choose_color_A, axis=1)
    colors = jp.vstack([color_A, color_B])
    checkered_pattern_colors = jp.take(colors, choose_color_A, axis=0)
    return checkered_pattern_colors
