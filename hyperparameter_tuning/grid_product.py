from itertools import product


class GridProductGenerator():
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def product_grid(self):
        param_values=list(self.param_grid.values())
        param_names=list(self.param_grid.keys())
        param_combos=[]

        for param_value in product(*param_values):
            param_combos.append(dict(zip(param_names, param_value)))

        return param_combos
