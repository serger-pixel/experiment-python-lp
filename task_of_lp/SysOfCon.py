import numpy as np

# Класс системы ограничений
class SysOfCon:
    def __init__(self, a, b, constraint):
        self.cntRows = a.shape[0]
        self.cntCols = a.shape[1] + b.shape[1] + constraint.shape[1]
        self.system = np.zeros((self.cntRows, self.cntCols))
        self.system[:a.shape[0], :a.shape[1]] = a
        self.system[self.cntRows:, :a.shape[1]] = b
        self.system[self.cntRows:, a.shape[1] + b.shape[1]:] = constraint