from optax import scale_by_backtracking_linesearch
from optax import scale_by_zoom_linesearch


def wolfe_linesearch(max_line_steps, verbose):
    # Wolfe criterion with zoom line search.
    kwargs = {"verbose": verbose, "initial_guess_strategy": "one"}
    return scale_by_zoom_linesearch(max_line_steps, **kwargs)


def armijo_linesearch(max_line_steps, verbose):
    # Armijo criterion with backtracking line search.
    kwargs = {"verbose": verbose, "store_grad": True, "slope_rtol": 1e-5}
    return scale_by_backtracking_linesearch(max_line_steps, **kwargs)


def LineSearch(max_line_steps, criterion, verbose):
    if criterion == "wolfe":
        return wolfe_linesearch(max_line_steps, verbose)
    if criterion == "armijo":
        return armijo_linesearch(max_line_steps, verbose)
    raise ValueError("`criterion` must be 'armijo' or 'wolfe'.")
