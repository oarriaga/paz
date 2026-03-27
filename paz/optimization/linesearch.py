from optax import scale_by_backtracking_linesearch
from optax import scale_by_zoom_linesearch


def zoom_linesearch(max_line_steps, verbose):
    kwargs = {"verbose": verbose, "initial_guess_strategy": "one"}
    return scale_by_zoom_linesearch(max_line_steps, **kwargs)


def backtracking_linesearch(max_line_steps, verbose):
    kwargs = {"verbose": verbose, "store_grad": True, "slope_rtol": 1e-5}
    return scale_by_backtracking_linesearch(max_line_steps, **kwargs)


def LineSearch(max_line_steps, wolfe_criterion, verbose):
    if wolfe_criterion:
        return zoom_linesearch(max_line_steps, verbose)
    return backtracking_linesearch(max_line_steps, verbose)
