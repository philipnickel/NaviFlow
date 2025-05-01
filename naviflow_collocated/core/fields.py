# naviflow_collocated/core/fields.py

import numpy as np


class CellField:
    def __init__(self, n_cells: int, name: str = ""):
        self.name = name
        self.values = np.zeros(n_cells, dtype=np.float64)

    def set_value(self, value: float):
        self.values.fill(value)

    def copy(self):
        new_field = CellField(len(self.values), name=self.name + "_copy")
        new_field.values[:] = self.values[:]
        return new_field

    def norm(self):
        return np.linalg.norm(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = val


class CellVectorField:
    def __init__(self, n_cells: int, name: str = ""):
        self.name = name
        self.values = np.zeros((n_cells, 2), dtype=np.float64)

    def set_value(self, value: tuple):
        self.values[:, 0] = value[0]
        self.values[:, 1] = value[1]

    def copy(self):
        new_field = CellVectorField(len(self.values), name=self.name + "_copy")
        new_field.values[:, :] = self.values[:, :]
        return new_field

    def norm(self):
        return np.linalg.norm(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = val
