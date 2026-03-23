import numpy as np

from task_of_lp.Condtition import Condition
from task_of_lp.Task import Task


class BigMMethod:
    def __init__(self, system, free_variables, constraints,
                 objective_function, M, task):
        self.system = system
        self.free_variables = free_variables
        self.constraints = constraints
        self.objective_function = objective_function
        self.M = M
        self.task = task


        self.table = None
        self.objective_function_m = None
        self.cntRow = None
        self.cntCol = None

        self.basis = None
        self.slacks = None
        self.surpluses = None
        self.artificials = None

    def create_table(self):
        self.cntRow = self.system.shape[0]
        self.cntCol = self.system.shape[1]
        self.table = self.system[:]


        # Списки базисных и искусственных переменных
        self.basis = np.zeros(self.cntRow, dtype=int)
        self.slacks = []
        self.surpluses = []
        self.artificials = []


        for row in range(self.cntRow):
            # Добавление доп. переменной
            if self.constraints[row] == Condition.LessEq:
                new_col = np.zeros(self.cntRow, dtype = float)
                new_col[row] = 1
                np.column_stack((self.table, new_col))
                self.basis[row] = (self.cntCol-1)
                self.slacks.append(self.cntCol-1)
                self.cntCol += 1


            # Добавление доп. и иск. переменных
            elif self.constraints[row] == Condition.GreaterEq:
                new_col = np.zeros(self.cntRow, dtype=float)
                new_col[row] = -1
                self.table = np.column_stack((self.table, new_col))
                self.surpluses.append(self.cntCol - 1)
                self.cntCol += 1


                new_col = np.zeros(self.cntRow, dtype=float)
                new_col[row] = 1
                self.table = np.column_stack((self.table, new_col))
                self.basis[row] = (self.cntCol-1)
                self.artificials.append(self.cntCol - 1)
                self.cntCol += 1

            else:
                new_col = np.zeros(self.cntRow, dtype=float)
                new_col[row] = 1
                self.table = np.column_stack((self.table, new_col))
                self.basis[row] = (self.cntCol - 1)
                self.artificials.append(self.cntCol - 1)
                self.cntCol += 1



        # Рассширение коэф. целевой функции
        previous_len = self.objective_function.shape[0]
        current_len = self.table.shape[1]
        new_var = np.zeros(current_len - previous_len, dtype=float)
        self.objective_function = np.hstack((self.objective_function, new_var))

        if self.task == Task.Min:
            self.objective_function *= -1

        # Коэффициенты целевой функции, которые зависят M
        self.objective_function_m = np.zeros(self.table.shape[1], dtype=float)
        sum_free_var_m = 0
        for row in range(self.cntRow):
            for artificial in self.artificials:
                if self.table[row, artificial] == 1:
                    self.objective_function_m -= self.table[row]
                    self.objective_function_m[artificial] = 0
                    sum_free_var_m += self.free_variables[row]

        if self.task == Task.Max:
            self.objective_function_m *= -1


        self.objective_function *= -1
        self.objective_function_m *= -1


        self.table = np.vstack((self.table, self.objective_function))
        self.table = np.vstack((self.table, self.objective_function_m))
        self.free_variables = np.hstack((self.free_variables, [0]))
        self.free_variables = np.hstack((self.free_variables, sum_free_var_m))
        self.table = np.column_stack((self.table, self.free_variables))



    def find_column(self, index):
        last_row = self.table[index]
        min_val = np.min(last_row)

        if min_val >= 0:
            return None

        return np.argmin(last_row)

    def find_row(self, column):
        s = []
        for row in range(self.cntRow):
            if self.table[row, column] > 1e-10:
                s_i = self.table[row, -1] / self.table[row, column]
                s.append((s_i, row))

        if not s:
            return None

        return min(s)[1]

    def step(self, row, column):
        pivot_element = self.table[row, column]
        self.table[row] /= pivot_element

        for i in range(self.cntRow):
            if i != row:
                current_pivot = self.table[i, column]
                self.table[i] -= current_pivot * self.table[row]

        self.basis[row] = column

    def solve(self, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("Продолжение по 1-ой оценочной строчке")
            print("=" * 60)
            self.print_table()

        while True:
            # Поиск ведущего столбца
            column = self.find_column(self.cntRow - 2)
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
            if verbose:
                self.print_table()

        solution = np.zeros(self.n_vars + self.n_b)
        for i in range(self.n_b):
            solution[self.basis[i]] = self.table[i, -1]

        z_value = self.table[-1, -1]
        return True, solution, z_value


    def solve_M(self, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("По 2-ой оценочной строчке (коэф. при M)")
            print("=" * 60)
            self.print_table()

        while True:
            column = self.find_column(self.cntCol - 1)
            if column is None:
                for basis in range(self.cntRow):
                    for artificial in self.artificials:
                        if artificial == basis:
                            print("\n Система не совместна")
                        else:
                            return self.solve()
                break

            row = self.find_row(column)
            if row is None:
                if verbose:
                    print("\nЗадача не ограничена!")
                return False, None, None

            # Выполнение шага
            if verbose:
                print(f"\n--- Итерация {self.iterations + 1} ---")
                print(f"Разрешающий столбец: {column + 1}")
                print(f"Разрешающая строка: {row + 1}")
            self.step(row, column)
            if verbose:
                self.print_table()

        solution = np.zeros(self.cntCol)
        for i in range(self.cntRow):
            solution[self.basis[i]] = self.table[i, -1]

        z_value = self.table[self.cntCol - 2, -1]

        return True, solution, z_value

    def print_table(self):
        print("\nСимплекс-таблица:")
        print("-" * 60)


        header = "Базис | "
        for i in range(self.cntCol):
            header += f"     x{i + 1}"
        header += "|  b  "
        print(header)
        print("-" * 60)


        for i in range(self.cntRow):
            row = f"  x{self.basis[i] :3} | "
            for j in range(self.cntCol):
                row += f"{self.table[i, j]:6.2f} "
            row += f"| {self.table[i, -1]:6.2f}"
            print(row)

        print("-" * 60)

        row = "  Z   | "
        for j in range(self.cntCol):
            row += f"{self.table[-2, j]:6.2f} "
        row += f"| {self.table[-2, -1]:6.2f}"
        print(row)
        print("-" * 60)

        row = "  Zm   | "
        for j in range(self.cntCol):
            row += f"{self.table[-1, j]:6.2f} "
        row += f"| {self.table[-1, -1]:6.2f}"
        print(row)
        print("-" * 60)







