from grid_product import GridProductGenerator


class HyperparameterTuning:
    def __init__(self,model_name, model_function):


    def tune(self):
        grid_generator = GridProductGenerator(self.param_grid)
        combos=grid_generator.product_grid()
        tuning_results=[]
        for combo in combos:


