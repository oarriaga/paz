import numpy as np


class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass


class DISALLOWED_OBJ(object):
    pass


DISALLOWED = DISALLOWED_OBJ()
DISALLOWED_PRINTVAL = "D"


def get_cover_matrix(shape):
    """Returns the initialized row and column cover matrix.

    # Arguments
        shape: Tuple. Shape of the cover matrix.
    """
    row_covered = np.zeros(shape, dtype=bool)
    col_covered = np.zeros(shape, dtype=bool)
    return row_covered, col_covered


def find_uncovered_zero(n, cost_matrix, row_covered, col_covered, i0, j0):
    row = -1
    col = -1
    done = False
    for row_arg in range(i0, n):
        for col_arg in range(j0, n):
            if (cost_matrix[row_arg][col_arg] == 0) and \
                    (not row_covered[row_arg]) and \
                    (not col_covered[col_arg]):
                row = row_arg
                col = col_arg
                done = True
        if done:
            break
    return (row, col)


def find_star_in_row(n, row_arg, marked):
    col = -1
    for col_arg in range(n):
        if marked[row_arg][col_arg] == 1:
            col = col_arg
            break
    return col


def find_star_in_col(n, col_arg, marked):
    row = -1
    for row_arg in range(n):
        if marked[row_arg][col_arg] == 1:
            row = row_arg
            break
    return row


def find_prime_in_row(n, row_arg, marked):
    col = -1
    for col_arg in range(n):
        if marked[row_arg][col_arg] == 2:
            col = col_arg
            break
    return col


def get_min_value(series):
    values = []
    for x in series:
        if type(x) is not type(DISALLOWED):
            values.append(x)
    if len(values) == 0:
        raise UnsolvableMatrix("One row is entirely DISALLOWED.")
    min_value = np.min(values)
    return min_value


def find_smallest_uncovered(n, row_covered, col_covered, cost_matrix):
    minval = np.inf
    for i in range(n):
        for j in range(n):
            if (not row_covered[i]) and (not col_covered[j]):
                if cost_matrix[i][j] is not DISALLOWED and \
                        minval > cost_matrix[i][j]:
                    minval = cost_matrix[i][j]
    return minval
