import numpy as np

from simplex.Simplex import SimplexMethod
from task_of_lp.TableService import TableService


class DualSimplexMethod:
    def __init__(self, table, basis):
        self.table = table
        self.cnt_row = table.shape[0]
        self.cnt_col = table.shape[1]
        self.basis = basis
        self.simplex = SimplexMethod(table, basis)
        self.service = TableService(table, basis)

    def find_row(self):
        rows = []

        for row in range(self.basis.shape[0]):
            if self.table[row, -1] < 0:
                for col in range(self.table.shape[1] - 1):
                    if self.table[row, col] < 0:
                        rows.append((self.table[row, -1], row))
                        break
        if not rows:
            return None

        return min(rows)[1]

    def find_col(self, row):
        dual_values = []
        for col in range(self.table.shape[1] - 1):
            if self.table[row, col] < 0:
                dual_value = abs(self.table[-1, col] / self.table[row, col])
                dual_values.append((dual_value, col))

        min_col = min(dual_values)[1]
        return min_col


    def step(self, row, col):
        pivot_element = self.table[row, col]
        self.table[row] /= pivot_element

        for current_row in range(self.cnt_row):
            if current_row != row:
                current_pivot = self.table[current_row, col]
                self.table[current_row] -= current_pivot * self.table[row]

        self.basis[row] = col

    def check_b(self):
        for row in range(self.basis.shape[0]):
            if self.table[row, -1] < 0:
                return True
        return False

    def solve(self):

        print("=" * 60)
        self.service.print_table(0)
        while True:
            if not self.check_b():
                return self.simplex.solve(0, -1, True)

            row = self.find_row()
            if row is None:
                print("\nСистема линейных уравнений несовместна")
                return False, None, None
            column = self.find_col(row)

            self.step(row, column)


            self.service.print_pivot(row, column)
            self.service.print_table(0)