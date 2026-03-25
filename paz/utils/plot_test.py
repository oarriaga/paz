import paz.utils.plot as utils_plot


def test_plot_module_imports_from_utils_package():
    assert callable(utils_plot.save)
    assert callable(utils_plot.subplots)
