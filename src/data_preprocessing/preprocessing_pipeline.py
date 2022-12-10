import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Deal with numeric variable standardization, categorical encoding and NA handling

class PreprocessingPipeline():
    def __init__(self, id_col, response_variables, num_variables, cat_variables):
        self.id_col = id_col
        self.response_variables = response_variables
        self.num_variables = num_variables
        self.cat_variables = cat_variables
        self.all_features = self.num_variables + self.cat_variables

    def create_preprocessing_pipeline(self):
        numeric_processing = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        categorical_processing = Pipeline(
            steps=[("cat_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]
        )

        preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ("num_variables", numeric_processing, self.num_variables),
                ("cat_variables", categorical_processing, self.cat_variables)
            ]
        )

        return preprocessing_pipeline