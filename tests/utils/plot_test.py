import numpy as np

import paz.utils.plot as utils_plot


def test_corner_draws_a_single_variable():
    samples = np.linspace(0.0, 1.0, 50)
    figure, axes = utils_plot.corner({"a": samples})
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_xlabel() == "a"
    assert len(axes[0, 0].patches) == 30
    utils_plot.plt.close(figure)
