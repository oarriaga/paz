from paz.processors import Munkres
from paz.backend import munkres
import pytest


DISALLOWED = munkres.DISALLOWED_OBJ()


@pytest.fixture
def cost_matrices():
    matrices = [
        # Square
        ([[400, 150, 400],
            [400, 450, 600],
            [300, 225, 300]],
            850),

        # Rectangular variant
        ([[400, 150, 400, 1],
            [400, 450, 600, 2],
            [300, 225, 300, 3]],
            452),

        # Square variant with floating point value
        ([[10.1, 10.2,  8.3],
            [9.4,  8.5,  1.6],
            [9.7,  7.8,  4.9]],
            19.5),

        # Rectangular variant with floating point value
        ([[10.01, 10.02,  8.03, 11.04],
            [9.05,  8.06,  1.07, 1.08],
            [9.09,  7.1,  4.11, 10.12]],
            15.2),

        # Rectangular with DISALLOWED
        ([[4, 5, 6, DISALLOWED],
            [1, 9, 12, 11],
            [DISALLOWED, 5, 4, DISALLOWED],
            [12, 12, 12, 10]],
            20),

        # Rectangular variant with DISALLOWED and floating point value
        ([[4.001, 5.002, 6.003, DISALLOWED],
            [1.004, 9.005, 12.006, 11.007],
            [DISALLOWED, 5.008, 4.009, DISALLOWED],
            [12.01, 12.011, 12.012, 10.013]],
            20.028),

        # DISALLOWED to force pairings
        ([[1, DISALLOWED, DISALLOWED, DISALLOWED],
            [DISALLOWED, 2, DISALLOWED, DISALLOWED],
            [DISALLOWED, DISALLOWED, 3, DISALLOWED],
            [DISALLOWED, DISALLOWED, DISALLOWED, 4]],
            10),

        # DISALLOWED to force pairings with floating point value
        ([[1.1, DISALLOWED, DISALLOWED, DISALLOWED],
            [DISALLOWED, 2.2, DISALLOWED, DISALLOWED],
            [DISALLOWED, DISALLOWED, 3.3, DISALLOWED],
            [DISALLOWED, DISALLOWED, DISALLOWED, 4.4]],
            11.0)]
    return (matrices)


@pytest.fixture
def rectangular_cost_matrix():
    M = [[400, 150, 400, 1],
         [400, 450, 600, 2],
         [300, 225, 300, 3]]
    return M


@pytest.fixture
def marked_matrix():
    marked = [[0, 0, 1, 1],
              [0, 1, 0, 2],
              [0, 0, 1, 0]]
    return marked


@pytest.fixture
def row_covered():
    return [True, False, False, False]


@pytest.fixture
def col_covered():
    return [True, False, False, False]


def test_matrix_cost(cost_matrices):
    m = Munkres()
    for cost_matrix, expected_total in cost_matrices:
        indexes = m.compute(cost_matrix)
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r][c]
            total_cost += x
        assert expected_total == total_cost


@pytest.mark.parametrize("expected_min_value", [1])
def test_get_min_value(rectangular_cost_matrix, expected_min_value):
    min_value = munkres.get_min_value(rectangular_cost_matrix[0])
    assert (min_value == expected_min_value)
