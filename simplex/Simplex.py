import numpy as np
from typing import List


class SimplexMethod:
    def __init__(self, z, a, b):

        self.n_rows = None
        self.n_cols = None

        self.z = np.array(z, dtype=float)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)

        self.n_vars = self.a.shape[1]
        self.n_b = self.b.shape[0]

        self.table = self.create_table()
        self.basis = np.arange(self.n_vars, self.n_vars + self.n_b)

        self.iterations = 0

    def create_table(self):

        self.n_rows = self.n_b + 1
        self.n_cols = self.n_vars + self.n_b + 1
        table = np.zeros((self.n_rows, self.n_cols))

        table[:self.n_b, :self.n_vars] = self.a
        table[:self.n_b, -1] = self.b

        table[-1, :self.n_vars] = self.z

        table[:self.n_b, self.n_vars:self.n_vars + self.n_b] = np.eye(self.n_b)

        return table

    def find_column(self):
        last_row = self.table[-1, :-1]
        min_val = np.min(last_row)

        if min_val >= 0:
            return None

        return np.argmin(last_row)

    def find_row(self, column):
        s = []

        for row in range(self.n_b):
            if self.table[row, column] > 1e-10:
                s_i = self.table[row, -1] / self.table[row, column]
                s.append((s_i, row))

        if not s:
            return None

        return min(s)[1]

    def step(self, row, column):
        pivot_element = self.table[row, column]
        self.table[row] /= pivot_element

        for i in range(self.n_rows):
            if i != row:
                current_pivot = self.table[i, column]
                self.table[i] -= current_pivot * self.table[row]

        self.basis[row] = column

    def solve(self, verbose: bool = False):
        if verbose:
            print("=" * 60)
            print("НАЧАЛО РЕШЕНИЯ СИМПЛЕКС-МЕТОДОМ")
            print("=" * 60)
            self.print_table()

        while True:
            # Поиск ведущего столбца
            column = self.find_column()
            if column is None:
                if verbose:
                    print("\nОптимальное решение найдено!")
                break

            row = self.find_row(column)
            if row is None:
                if verbose:
                    print("\nЗадача неограничена!")
                return False, None, None

            # Выполнение шага
            if verbose:
                print(f"\n--- Итерация {self.iterations + 1} ---")
                print(f"Разрешающий столбец: {column + 1}")
                print(f"Разрешающая строка: {row + 1}")
            self.step(row, column)
            self.iterations += 1
            if verbose:
                self.print_table()

        solution = np.zeros(self.n_vars + self.n_b)
        for i in range(self.n_b):
            solution[self.basis[i]] = self.table[i, -1]

        z_value = self.table[-1, -1]

        return True, solution, z_value

    def print_table(self):
        print("\nСимплекс-таблица:")
        print("-" * 60)


        header = "Базис | "
        for i in range(self.n_vars + self.n_b):
            header += f"     x{i + 1}"
        header += "|  b  "
        print(header)
        print("-" * 60)

        for i in range(self.n_b):
            row = f"  x{self.basis[i] + 1:3} | "
            for j in range(self.n_vars + self.n_b):
                row += f"{self.table[i, j]:6.2f} "
            row += f"| {self.table[i, -1]:6.2f}"
            print(row)

        print("-" * 60)

        row = "  Z   | "
        for j in range(self.n_vars + self.n_b):
            row += f"{self.table[-1, j]:6.2f} "
        row += f"| {self.table[-1, -1]:6.2f}"
        print(row)
        print("-" * 60)


