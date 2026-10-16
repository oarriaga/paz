import jax
import jax.numpy as jp

import paz
from paz.backend import geometric


def split_keys(seed, num_keys):
    return jax.random.split(jax.random.PRNGKey(seed), num_keys)


def test_generate_crop_dimensions_draws_width_and_height_independently():
    generate = paz.lock(geometric.generate_crop_dimensions, 100, 100)
    widths, heights = jax.vmap(generate)(split_keys(0, 64))
    assert not bool(jp.all(widths == heights))
    generate = paz.lock(geometric.generate_crop_dimensions, 100, 60)
    widths, heights = jax.vmap(generate)(split_keys(1, 64))
    assert 30 <= int(widths.min()) and 60 <= int(widths.max()) < 100
    assert 18 <= int(heights.min()) and int(heights.max()) < 60


def test_build_crop_region_draws_x_and_y_independently():
    build = paz.lock(geometric.build_crop_region, 10, 10, 100, 100)
    regions = jax.vmap(build)(split_keys(2, 64))
    assert not bool(jp.all(regions[:, 0] == regions[:, 1]))
    build = paz.lock(geometric.build_crop_region, 20, 10, 100, 60)
    regions = jax.vmap(build)(split_keys(3, 64))
    assert int(regions[:, :2].min()) >= 0
    assert 50 <= int(regions[:, 0].max()) < 80
    assert int(regions[:, 1].max()) < 50
    sizes = regions[:, 2:] - regions[:, :2]
    assert bool(jp.all(sizes == jp.array([20, 10])))
