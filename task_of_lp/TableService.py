import numpy as np

from task_of_lp.Condtition import Condition
from task_of_lp.Task import Task


class TableService:
    def __init__(self):
        return self

    def __init__(self, table, basis):
        self.table = table
        self.basis = basis
        self.cnt_row, self.cnt_col = table.shape

    def create_table(self, system, free_variables, constraints,
                 objective_function, task):
        cnt_row = system.shape[0]
        cnt_col = system.shape[1]
        table = system[:]

        # Списки базисных и искусственных переменных
        basis = np.zeros(self.cnt_row, dtype=int)
        slacks = []
        surpluses = []
        artificials = []

        for row in range(cnt_row):
            # Добавление доп. переменной
            if constraints[row] == Condition.LessEq:
                new_col = np.zeros(cnt_row, dtype=float)
                new_col[row] = 1
                np.column_stack((table, new_col))
                basis[row] = (cnt_col - 1)
                slacks.append(cnt_col - 1)
                cnt_col += 1


            # Добавление доп. и иск. переменных
            elif constraints[row] == Condition.GreaterEq:
                new_col = np.zeros(cnt_row, dtype=float)
                new_col[row] = -1
                table = np.column_stack((table, new_col))
                cnt_col = table.shape[1]
                surpluses.append(cnt_col - 1)

                new_col = np.zeros(cnt_row, dtype=float)
                new_col[row] = 1
                table = np.column_stack((table, new_col))
                cnt_col = table.shape[1]
                basis[row] = (cnt_col - 1)
                artificials.append(cnt_col - 1)

            else:
                new_col = np.zeros(cnt_row, dtype=float)
                new_col[row] = 1
                table = np.column_stack((table, new_col))
                cnt_col += 1
                basis[row] = (cnt_col - 1)
                artificials.append(cnt_col - 1)

        # Рассширение коэф. целевой функции
        previous_len = objective_function.shape[0]
        current_len = table.shape[1]
        new_var = np.zeros(current_len - previous_len, dtype=float)
        objective_function = np.hstack((objective_function, new_var))

        if task == Task.Min:
            objective_function *= -1

        # Коэффициенты целевой функции, которые зависят M
        objective_function_m = np.zeros(table.shape[1], dtype=float)
        sum_free_var_m = 0
        for row in range(cnt_row):
            for artificial in artificials:
                if table[row, artificial] == 1:
                    objective_function_m -= table[row]
                    objective_function_m[artificial] = 0
                    sum_free_var_m -= free_variables[row]

        if task == Task.Max:
            objective_function_m *= -1

        objective_function *= -1
        objective_function_m *= -1

        table = np.vstack((table, objective_function))
        table = np.vstack((table, objective_function_m))
        free_variables = np.hstack((free_variables, [0]))
        free_variables = np.hstack((free_variables, sum_free_var_m))
        table = np.column_stack((table, free_variables))
        cnt_row = table.shape[0]
        cnt_col = table.shape[1]

        return table, basis, slacks, surpluses, artificials


    def print_pivot(self, row, column):
        print(f"\n------------------------")
        print(f"Разрешающий столбец: {column + 1}")
        print(f"Разрешающая строка: {row + 1}")


    def print_row(self, ind, row_header):
        row = row_header
        for j in range(self.cnt_col):
            row += f"{self.table[ind, j]:6.2f} "
        print(row)

    def print_table(self, without):
        print("\nСимплекс-таблица:")
        print("-" * 60)
        header = "Базис | "
        for i in range(self.cnt_col - 1):
            header += f"     x{i}"
        header += "|  b  "
        print(header)
        print("-" * 60)

        for i in range(self.basis.shape[0]):
            row = f"  x {self.basis[i]} | "
            for j in range(self.cnt_col):
                row += f"{self.table[i, j]:6.2f} "
            print(row)
        print("-" * 60)

        for i in range(self.cnt_row - self.basis.shape[0] - without):
            self.print_row(-i, "Z")