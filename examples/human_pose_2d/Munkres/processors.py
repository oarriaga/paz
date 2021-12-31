from paz import processors as pr
import backend as B
import numpy as np


class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass


class PadMatrix(pr.Processor):
    def __init__(self):
        super(PadMatrix, self).__init__()

    def call(self, cost_matrix):
        return B.pad_matrix(cost_matrix)


class GenerateCoverMatrix(pr.Processor):
    def __init__(self):
        super(GenerateCoverMatrix, self).__init__()

    def call(self, shape):
        return B.generate_cover_matrix(shape)


class GenerateZerosMatrix(pr.Processor):
    def __init__(self):
        super(GenerateZerosMatrix, self).__init__()

    def call(self, shape, dtype=np.int):
        return B.generate_zeros_matrix(shape, dtype)


class GetMinValue(pr.Processor):
    def __init__(self):
        super(GetMinValue, self).__init__()

    def call(self, series):
        return B.get_min_value(series)


class FindUncoveredZero(pr.Processor):
    def __init__(self):
        super(FindUncoveredZero, self).__init__()

    def call(self, n, cost_matrix, row_covered, col_covered, i0, j0):
        return B.find_uncovered_zero(n, cost_matrix, row_covered,
                                     col_covered, i0, j0)


class FindStarInRow(pr.Processor):
    def __init__(self):
        super(FindStarInRow, self).__init__()

    def call(self, n, row_arg, marked):
        return B.find_star_in_row(n, row_arg, marked)


class FindPrimeInRow(pr.Processor):
    def __init__(self):
        super(FindPrimeInRow, self).__init__()

    def call(self, n, row_arg, marked):
        return B.find_prime_in_row(n, row_arg, marked)


class FindStarInColumn(pr.Processor):
    def __init__(self):
        super(FindStarInColumn, self).__init__()

    def call(self, n, col_arg, marked):
        return B.find_star_in_col(n, col_arg, marked)


class FindSmallestUncovered(pr.Processor):
    def __init__(self):
        super(FindSmallestUncovered, self).__init__()

    def call(self, n, row_covered, col_covered, cost_matrix):
        return B.find_smallest_uncovered(n, row_covered,
                                         col_covered, cost_matrix)
