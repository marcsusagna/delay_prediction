import pandas as pd

# Deal with numeric variable standardization, categorical encoding and NA handling

class PreprocessingPipeline():
    def __init__(self, id_col, response_variables, num_variables, cat_variables):
        self.id_col = id_col
        self.response_variables = response_variables
        self.num_variables = num_variables
        self.cat_variables = cat_variables