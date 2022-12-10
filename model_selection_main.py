# To be run from project root directory
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, KFold

from sklearn.base import clone

from src.data_preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.data_preparation import constants as prep_constants

from src.model import utils as model_utils
from src.model import constants as model_constants

df_train = pd.read_parquet(prep_constants.CLEAN_PATH+"train.parquet")

## Candidate model definition:

# Define preprocessing pipeline
pipe_wrapper = PreprocessingPipeline(
    id_col=model_constants.ID_COL,
    response_variables=model_constants.RESPONSE_VARIABLES,
    num_variables=model_constants.NUMERIC_VARIABLES,
    cat_variables=model_constants.CATEGORICAL_VARIABLES
)
preprocessing_pipeline = pipe_wrapper.create_preprocessing_pipeline()

# Extend pipeline until the prediction step
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", LogisticRegression())
])

## Model selection:

# Run CV
X_train, y_train = model_utils.create_X_and_y(df_train, pipe_wrapper, "binary")
static_folds = KFold(n_splits=5)
cv_scores_log_reg = cross_val_score(log_reg_pipeline, X_train, y_train, cv=static_folds)

# Pick best model:
untrained_chosen_model = clone(log_reg_pipeline)
cv_scores_best_model = cv_scores_log_reg

df_validation = pd.read_parquet(prep_constants.CLEAN_PATH+"validation.parquet")

chosen_model_blueprint = model_utils.create_model_blueprint(
    model_version_tag=model_constants.MODEL_NEW_VERSION,
    untrained_chosen_model=untrained_chosen_model,
    pipeline_wrapper=pipe_wrapper,
    cv_scores=cv_scores_best_model,
    df_train=df_train,
    df_validation=df_validation,
)

# Current benchmark model scores:
benchmark_model = model_utils.fetch_model_blueprint_from_registry(model_constants.MODEL_CURRENT_VERSION)
print("Benchmark model performance")
print("cv_score", benchmark_model["metrics"]["cv_score"].mean())
print("validation_score", benchmark_model["metrics"]["cv_score"].mean())

# New model scores:
print("New model performance")
print("cv_score", chosen_model_blueprint["metrics"]["cv_score"].mean())
print("validation_score", chosen_model_blueprint["metrics"]["cv_score"].mean())


model_utils.put_model_blueprint_on_registry(chosen_model_blueprint)


