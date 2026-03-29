import numpy as np

from task_of_lp.TableService import TableService


class SimplexMethod:
    def __init__(self, table, basis):
        self.table = table
        self.cnt_row = table.shape[0]
        self.cnt_col = table.shape[1]
        self.basis = basis
        self.service = TableService(table, basis)

    def find_column(self, ind):
        last_row = self.table[ind, :-1]
        min_val = np.min(last_row)

        if min_val >= 0:
            return None

        return np.argmin(last_row)


    def find_row(self, column):
        simplex_values = []

        for row in range(self.basis.shape[0]):
            if self.table[row, column] > 1e-10:
                simplex_i = self.table[row, -1] / self.table[row, column]
                simplex_values.append((simplex_i, row))

        if not simplex_values:
            return None

        return min(simplex_values)[1]


    def step(self, row, column):
        pivot_element = self.table[row, column]
        self.table[row] /= pivot_element

        for current_row in range(self.cnt_row):
            if current_row != row:
                current_pivot = self.table[current_row, column]
                self.table[current_row] -= current_pivot * self.table[row]

        self.basis[row] = column

    def solve(self, without, ind, is_optimal_header):
        """
        :param without: кол-во невыводящих строк с конца
        :param ind: индекс строки, по которой идет поиск столбца
        :param is_optimal_header: флаг на отображения заголовка оптимального решения
        :return: решение задачи, строку под индексом ind
        """
        print("=" * 60)
        self.service.print_table(without)

        while True:
            column = self.find_column(ind)
            if column is None:
                if is_optimal_header:
                    print("\nОптимальное решение найдено!")
                solution = np.zeros(self.cnt_col)
                for i in range(self.basis.shape[0]):
                    solution[self.basis[i]] = self.table[i, -1]
                z = self.table[ind]
                return True, solution, z


            row = self.find_row(column)
            if row is None:
                print("\nЗадача неограничена!")
                return False, None, None

            self.service.print_pivot(row, column)
            self.step(row, column)
            self.service.print_table(without)





