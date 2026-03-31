from optax import scale_by_backtracking_linesearch
from optax import scale_by_zoom_linesearch


def wolfe_linesearch(max_line_steps):
    kwargs = {"verbose": False, "initial_guess_strategy": "one"}
    return scale_by_zoom_linesearch(max_line_steps, **kwargs)


def armijo_linesearch(max_line_steps):
    kwargs = {"verbose": False, "store_grad": True, "slope_rtol": 1e-5}
    return scale_by_backtracking_linesearch(max_line_steps, **kwargs)


def LineSearch(max_line_steps, criterion):
    if criterion == "wolfe":
        return wolfe_linesearch(max_line_steps)
    if criterion == "armijo":
        return armijo_linesearch(max_line_steps)
    raise ValueError("`criterion` must be 'armijo' or 'wolfe'.")
