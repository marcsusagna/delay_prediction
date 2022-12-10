import pickle
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

from sklearn.base import clone

from src.data_preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.data_preprocessing import constants

df_train = pd.read_parquet("data/clean/train.parquet")

# Define preprocessing pipeline
pipe_wrapper = PreprocessingPipeline(
    id_col=constants.ID_COL,
    response_variables=constants.RESPONSE_VARIABLES,
    num_variables=constants.NUMERIC_VARIABLES,
    cat_variables=constants.CATEGORICAL_VARIABLES
)
preprocessing_pipeline = pipe_wrapper.create_preprocessing_pipeline()

# Define pipeline up to prediction
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", LogisticRegression())
])

# Run CV for model selection
X_train = df_train.loc[:, pipe_wrapper.id_col + pipe_wrapper.all_features]
y_train = df_train.loc[:, pipe_wrapper.response_variables[:1]]

static_folds = KFold(n_splits=5)
cv_scores_log_reg = cross_val_score(log_reg_pipeline, X_train, y_train, cv=static_folds)

# Pick best model:
chosen_model = log_reg_pipeline

# Store model configuration for model registry (deployment and improvement)
untrained_chosen_model = clone(chosen_model)
cv_scores_chosen_model = cv_scores_log_reg

# Train model on all train set and obtain validation score
chosen_model.fit(X_train, y_train)

df_validation = pd.read_parquet("data/clean/validation.parquet")

X_validation = df_validation.loc[:, pipe_wrapper.id_col + pipe_wrapper.all_features]
y_validation = df_validation.loc[:, pipe_wrapper.response_variables[:1]]

val_score = accuracy_score(y_validation, chosen_model.predict(X_validation))

# Store information of chosen model in model registry
dict_for_model_registry = {
    "model": {
        "model_version": "0.0.1",
        "pipe_wrapper": pipe_wrapper,
        "untrained_model": untrained_chosen_model
    },
    "metrics": {
        "cv_scores": cv_scores_chosen_model,
        "validation_score": val_score
    }
}

model_blueprint_path = "model_registry/v0.0.1/blueprint.pkl"

os.makedirs(os.path.dirname(model_blueprint_path), exist_ok=True)
pickle.dump(dict_for_model_registry, open(model_blueprint_path, 'wb'))



