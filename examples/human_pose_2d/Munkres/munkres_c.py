import numpy as np
from paz import processors as pr
import processors as pe


class DISALLOWED_OBJ(object):
    pass


DISALLOWED = DISALLOWED_OBJ()
DISALLOWED_PRINTVAL = "D"


class Munkres(pr.Processor):
    def __init__(self, cost_matrix):
        super(Munkres, self).__init__()
        self.H, self.W = np.array(cost_matrix).shape[:2]
        self.pad_matrix = pe.PadMatrix()
        self.cost_matrix = self.pad_matrix(cost_matrix)
        self.n = len(self.cost_matrix)
        self.generate_zeros_matrix = pe.GenerateZerosMatrix()
        self.generate_cover_matrix = pe.GenerateCoverMatrix()
        self.marked = self.generate_zeros_matrix((self.n, self.n))
        self.path = self.generate_zeros_matrix((self.n * 2, self.n * 2))
        self.col_covered, self.row_covered = self.generate_cover_matrix(
                                                (self.n, 1))
        self.Z0_r = 0
        self.Z0_c = 0
        self.done = False
        self.find_uncovered_zero = pe.FindUncoveredZero()
        self.find_star_in_row = pe.FindStarInRow()
        self.find_star_in_col = pe.FindStarInColumn()
        self.find_prime_in_row = pe.FindPrimeInRow()
        # self.get_min_value = pe.GetMinValue()
        self.find_smallest_uncovered = pe.FindSmallestUncovered()
        self.unsolvable_matrix = pe.UnsolvableMatrix()
        self.steps = {1: self.step1,
                      2: self.step2,
                      3: self.step3,
                      4: self.step4,
                      5: self.step5,
                      6: self.step6}

    def compute(self):
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

    def convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def erase_primes(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

    def get_min_value(self, series):
        values = []
        for x in series:
            if x is not DISALLOWED:
                values.append(x)
        min_value = np.min(values)
        return min_value

    def step1(self):
        '''
        Find the smallest possible value from each row and subtract it to
        the respective row
        '''
        for row in range(self.n):
            min_value = self.get_min_value(self.cost_matrix[row])
            for col in range(self.n):
                if self.cost_matrix[row][col] is not DISALLOWED:
                    self.cost_matrix[row][col] = \
                        self.cost_matrix[row][col] - min_value
        return 2

    def step2(self):
        for row in range(self.n):
            for col in range(self.n):
                if (self.cost_matrix[row][col] == 0) and \
                    (not self.row_covered[row]) and \
                        (not self.col_covered[col]):
                    self.marked[row][col] = 1
                    self.row_covered[row] = True
                    self.col_covered[col] = True
                    break
        self.row_covered, self.col_covered = \
            self.generate_cover_matrix((self.n, 1))
        return 3

    def step3(self):
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

    def step4(self):
        done = False
        row = 0
        col = 0
        star_col = -1
        while not done:
            (row, col) = self.find_uncovered_zero(self.n, self.cost_matrix,
                                                  self.row_covered,
                                                  self.col_covered, row, col)

            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.find_star_in_row(self.n, row, self.marked)
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

    def step5(self):
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.find_star_in_col(self.n, path[count][1], self.marked)
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.find_prime_in_row(self.n, path[count][0],
                                             self.marked)
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.convert_path(path, count)
        self.row_covered, self.col_covered = \
            self.generate_cover_matrix((self.n, 1))
        self.erase_primes()
        return 3

    def step6(self):
        minval = self.find_smallest_uncovered(self.n, self.row_covered,
                                              self.col_covered,
                                              self.cost_matrix)
        events = 0
        for row_arg in range(self.n):
            for col_arg in range(self.n):
                if self.cost_matrix[row_arg][col_arg] is DISALLOWED:
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
            raise self.unsolvable_matrix("Matrix cannot be solved!")
        return 4


if __name__ == '__main__':

    matrices = [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         850),
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
         20)]

    for cost_matrix, expected_total in matrices:
        m = Munkres(cost_matrix)

        indexes = m.compute()
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r][c]
            total_cost += x
        assert expected_total == total_cost
