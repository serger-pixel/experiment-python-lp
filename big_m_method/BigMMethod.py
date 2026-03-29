import numpy as np

from simplex.Simplex import SimplexMethod
from task_of_lp.Condtition import Condition
from task_of_lp.TableService import TableService
from task_of_lp.Task import Task


class BigMMethod:
    def __init__(self, system, free_variables, constraints,
                 objective_function, task):
        self.system = system
        self.free_variables = free_variables
        self.constraints = constraints
        self.objective_function = objective_function
        self.task = task
        self.service = TableService()

        table_aspects = self.service.create_table(system, free_variables, constraints,
                 objective_function, task)

        self.table = table_aspects[0]
        self.objective_function_m = None
        self.cnt_row = None
        self.cnt_col = None

        self.basis = table_aspects[1]
        self.slacks = table_aspects[2]
        self.surpluses = table_aspects[4]
        self.artificials = table_aspects[5]

    def create_table(self):
        self.cnt_row = self.system.shape[0]
        self.cnt_col = self.system.shape[1]
        self.table = self.system[:]


        # Списки базисных и искусственных переменных
        self.basis = np.zeros(self.cnt_row, dtype=int)
        self.slacks = []
        self.surpluses = []
        self.artificials = []


        for row in range(self.cnt_row):
            # Добавление доп. переменной
            if self.constraints[row] == Condition.LessEq:
                new_col = np.zeros(self.cnt_row, dtype = float)
                new_col[row] = 1
                np.column_stack((self.table, new_col))
                self.basis[row] = (self.cnt_col-1)
                self.slacks.append(self.cnt_col-1)
                self.cnt_col += 1


            # Добавление доп. и иск. переменных
            elif self.constraints[row] == Condition.GreaterEq:
                new_col = np.zeros(self.cnt_row, dtype=float)
                new_col[row] = -1
                self.table = np.column_stack((self.table, new_col))
                self.cnt_col = self.table.shape[1]
                self.surpluses.append(self.cnt_col - 1)


                new_col = np.zeros(self.cnt_row, dtype=float)
                new_col[row] = 1
                self.table = np.column_stack((self.table, new_col))
                self.cnt_col = self.table.shape[1]
                self.basis[row] = (self.cnt_col-1)
                self.artificials.append(self.cnt_col - 1)

            else:
                new_col = np.zeros(self.cnt_row, dtype=float)
                new_col[row] = 1
                self.table = np.column_stack((self.table, new_col))
                self.cnt_col += 1
                self.basis[row] = (self.cnt_col - 1)
                self.artificials.append(self.cnt_col - 1)




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
        for row in range(self.cnt_row):
            for artificial in self.artificials:
                if self.table[row, artificial] == 1:
                    self.objective_function_m -= self.table[row]
                    self.objective_function_m[artificial] = 0
                    sum_free_var_m -= self.free_variables[row]

        if self.task == Task.Max:
            self.objective_function_m *= -1


        self.objective_function *= -1
        self.objective_function_m *= -1


        self.table = np.vstack((self.table, self.objective_function))
        self.table = np.vstack((self.table, self.objective_function_m))
        self.free_variables = np.hstack((self.free_variables, [0]))
        self.free_variables = np.hstack((self.free_variables, sum_free_var_m))
        self.table = np.column_stack((self.table, self.free_variables))
        self.cnt_row = self.table.shape[0]
        self.cnt_col = self.table.shape[1]

    def solve(self):
        self.create_table()
        simplex = SimplexMethod(self.table, self.basis)

        # Вывод с учетом всех строк
        without = 0

        has_solution, solution, z_m = simplex.solve(without,self.cnt_row - 1, False)
        if has_solution:
            for basis in self.basis:
                for artificial in self.artificials:
                    if basis == artificial:
                        print("Система несовместна в области допустимых решений")
                        return False, solution, z_m

            without = 1
            has_solution, solution, z_m = simplex.solve(without, self.cnt_row - 2, True)
            return has_solution, solution, z_m

        else:
            return  has_solution, solution, z_m














