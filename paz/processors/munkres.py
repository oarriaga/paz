import numpy as np
from ..abstract import Processor

from ..backend.munkres import UnsolvableMatrix
from ..backend.munkres import DISALLOWED_OBJ
from ..backend.munkres import get_cover_matrix
from ..backend.munkres import find_uncovered_zero
from ..backend.munkres import find_star_in_row
from ..backend.munkres import find_star_in_col
from ..backend.munkres import find_prime_in_row
from ..backend.munkres import get_min_value
from ..backend.munkres import find_smallest_uncovered

from ..backend.standard import pad_matrix

DISALLOWED = DISALLOWED_OBJ()


class Munkres(Processor):
    """
    Provides an implementation of the Munkres algorithm.

    # References
    https://brc2.com/the-algorithm-workshop/
    https://software.clapper.org/munkres/
    https://github.com/bmc/munkres
    """
    def __init__(self):
        super(Munkres, self).__init__()
        self.Z0_r = 0
        self.Z0_c = 0
        self.done = False
        self.steps = {1: self._step1,
                      2: self._step2,
                      3: self._step3,
                      4: self._step4,
                      5: self._step5,
                      6: self._step6}

    def compute(self, cost_matrix):
        self.H, self.W = np.array(cost_matrix).shape[:2]
        self.cost_matrix = pad_matrix(cost_matrix, padding='square')
        self.n = len(self.cost_matrix)
        self.marked = np.zeros((self.n, self.n), dtype=np.int)
        self.path = np.zeros((self.n * 2, self.n * 2), dtype=np.int)
        self.row_covered = np.zeros((self.n, 1), dtype=bool)
        self.col_covered = np.zeros((self.n, 1), dtype=bool)

        step = 1
        while not self.done:
            step_func = self.steps[step]
            step = step_func()
            if step == 7:
                break
        cost = []

        for row_arg in range(self.H):
            for col_arg in range(self.W):
                if self.marked[row_arg][col_arg]:
                    cost = cost + [(row_arg, col_arg)]
        return cost

    def _convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def _erase_primes(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

    def _step1(self):
        '''
        For each row of the matrix, find the smallest element and subtract
        it from every element in its row.  Go to Step 2.
        '''
        for row in range(self.n):
            min_value = get_min_value(self.cost_matrix[row])
            for col in range(self.n):
                if type(self.cost_matrix[row][col]) is not type(DISALLOWED):
                    self.cost_matrix[row][col] = \
                        self.cost_matrix[row][col] - min_value
        return 2

    def _step2(self):
        '''
        In the resulting matrix, look for a zero (Z). Star Z if there isn't
        a starred zero in its row or column. For each element in the matrix,
        repeat the process. Continue to Step 3.
        '''
        for row in range(self.n):
            for col in range(self.n):
                if (self.cost_matrix[row][col] == 0) and \
                    (not self.row_covered[row]) and \
                        (not self.col_covered[col]):
                    self.marked[row][col] = 1
                    self.row_covered[row] = True
                    self.col_covered[col] = True
                    break
        self.row_covered, self.col_covered = get_cover_matrix((self.n, 1))
        return 3

    def _step3(self):
        '''
        Cover each column containing a starred zero.  If K columns are covered,
        the starred zeros describe a complete set of unique assignments.
        In this case, Go to DONE, otherwise, Go to Step 4.
        '''
        count = 0
        for row in range(self.n):
            for col in range(self.n):
                if self.marked[row][col] == 1 and not self.col_covered[col]:
                    self.col_covered[col] = True
                    count = count + 1

        if count >= self.n:
            step = 7
        else:
            step = 4
        return step

    def _step4(self):
        '''
        Find a noncovered zero and prime it.  If there is no starred zero in
        the row containing this primed zero, Go to Step 5. Otherwise, cover
        this row and uncover the column containing the starred zero. Continue
        in this manner until there are no uncovered zeros left. Save the
        smallest uncovered value and Go to Step 6.
        '''
        done = False
        row = 0
        col = 0
        star_col = -1
        while not done:
            (row, col) = find_uncovered_zero(self.n, self.cost_matrix,
                                             self.row_covered,
                                             self.col_covered, row, col)

            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = find_star_in_row(self.n, row, self.marked)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5
        return step

    def _step5(self):
        '''
        Construct a series of alternating primed and starred zeros as follows.
        Let Z0 represent the uncovered primed zero found in Step 4. Let Z1
        denote the starred zero in the column of Z0 (if any). Let Z2 denote
        the primed zero in the row of Z1 (there will always be one). Continue
        until the series terminates at a primed zero that has no starred zero
        in its column. Unstar each starred zero of the series, star each primed
        zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3.
        '''
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = find_star_in_col(self.n, path[count][1], self.marked)
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = find_prime_in_row(self.n, path[count][0], self.marked)
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self._convert_path(path, count)
        self.row_covered, self.col_covered = get_cover_matrix((self.n, 1))
        self._erase_primes()
        return 3

    def _step6(self):
        '''
        Add the value found in Step 4 to every element of each covered row,
        and subtract it from every element of each uncovered column. Return
        to Step 4 without altering any stars, primes, or covered lines.
        '''
        minval = find_smallest_uncovered(self.n, self.row_covered,
                                         self.col_covered,
                                         self.cost_matrix)
        events = 0
        for row_arg in range(self.n):
            for col_arg in range(self.n):
                if type(self.cost_matrix[row_arg][col_arg]) \
                        is type(DISALLOWED):
                    continue
                if self.row_covered[row_arg]:
                    self.cost_matrix[row_arg][col_arg] += minval
                    events += 1
                if not self.col_covered[col_arg]:
                    self.cost_matrix[row_arg][col_arg] -= minval
                    events += 1
                if self.row_covered[row_arg] and not self.col_covered[col_arg]:
                    events -= 2
        if (events == 0):
            raise UnsolvableMatrix("Matrix cannot be solved!")
        return 4
