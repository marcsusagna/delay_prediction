# To be run from project root directory
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.base import clone

from src.data_preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.data_preparation import constants as prep_constants

from src.model import utils as model_utils
from src.model import constants as model_constants
from src.model.custom_estimators import zero_inflated_estimator, zero_inflated_log_estimator

df_train = pd.read_parquet(prep_constants.CLEAN_PATH+"train.parquet")

# Define preprocessing pipeline
pipe_wrapper = PreprocessingPipeline(
    id_col=model_constants.ID_COL,
    response_variables=model_constants.RESPONSE_VARIABLES,
    num_variables=model_constants.NUMERIC_VARIABLES,
    cat_variables=model_constants.CATEGORICAL_VARIABLES
)
preprocessing_pipeline = pipe_wrapper.create_preprocessing_pipeline()

### Model selection:

X_train, y_train = model_utils.create_X_and_y(df_train, pipe_wrapper)
static_folds = KFold(n_splits=5)
all_cvs = {}


## Candidate model definition:

# Candidate 1: Basic linear regression
lin_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", LinearRegression())
])
model_utils.add_cv_result(all_cvs, "lin_reg", lin_reg_pipeline, X_train, y_train, static_folds)

# Candidate 2: Basic zir model
zir_model = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", zero_inflated_estimator(LogisticRegression(max_iter=1000), LinearRegression()))
])
model_utils.add_cv_result(all_cvs, "zir", zir_model, X_train, y_train, static_folds)

# Candidate 3: Basic zir log model
zir_log_model = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", zero_inflated_log_estimator(LogisticRegression(max_iter=1000), LinearRegression()))
])
model_utils.add_cv_result(all_cvs, "zir_log", zir_model, X_train, y_train, static_folds)

# Candidate 4: Random forest
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessing_pipeline),
    ("model", RandomForestRegressor(n_estimators=10))
])
model_utils.add_cv_result(all_cvs, "rf", rf_model, X_train, y_train, static_folds)


# Pick best model:
untrained_chosen_model, cv_scores_best_model = model_utils.find_best_cv(all_cvs)

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
print("cv_score", benchmark_model["metrics"]["ml"]["cv_scores"].mean())
print("validation_regression_score", chosen_model_blueprint["metrics"]["ml"]["validation_regression_score"].mean())
print("validation_classification_score", chosen_model_blueprint["metrics"]["ml"]["validation_classification_score"].mean())

# New model scores:
print("New model performance")
print("cv_score", chosen_model_blueprint["metrics"]["ml"]["cv_scores"].mean())
print("validation_regression_score", chosen_model_blueprint["metrics"]["ml"]["validation_regression_score"].mean())
print("validation_classification_score", chosen_model_blueprint["metrics"]["ml"]["validation_classification_score"].mean())


model_utils.put_model_blueprint_on_registry(chosen_model_blueprint)


