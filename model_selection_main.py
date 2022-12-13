# Intended to be a jupyter notebook template for model exploration
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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

### Model selection:

X_train, y_train_reg, y_train_class = model_utils.create_X_and_y(df_train, pipe_wrapper)
static_folds = KFold(n_splits=5)

## Candidate model definition:

# Candidate 1: Basic linear regression AND a basic logistic regression
regression_pipeline = Pipeline(steps=[
    ("preprocessor", pipe_wrapper.create_preprocessing_pipeline()),
    ("model", LinearRegression())
])
cv_score_reg = cross_val_score(regression_pipeline, X_train, y_train_reg, cv=static_folds)

classification_pipeline = Pipeline(steps=[
    ("preprocessor", pipe_wrapper.create_preprocessing_pipeline()),
    ("model", LogisticRegression(max_iter=1000))
])
cv_score_class = cross_val_score(classification_pipeline, X_train, y_train_class, cv=static_folds)

print("Model regression CVs", cv_score_reg.mean())
print("Model classification CVs", cv_score_class.mean())

#untrained_chosen_model, cv_scores_best_model = model_utils.find_best_cv(all_cvs)

untrained_chosen_model = {
    "reg": regression_pipeline,
    "class": classification_pipeline
}
cv_scores = {"reg": cv_score_reg.mean(), "class": cv_score_class.mean()}

# Test best model on validation and save the model blueprint

df_validation = pd.read_parquet(prep_constants.CLEAN_PATH+"validation.parquet")

chosen_model_blueprint = model_utils.create_model_blueprint(
    model_version_tag=model_constants.MODEL_NEW_VERSION,
    untrained_chosen_model=untrained_chosen_model,
    pipeline_wrapper=pipe_wrapper,
    cv_score=cv_scores,
    df_train=df_train,
    df_validation=df_validation,
)

# Current benchmark model scores:
benchmark_model = model_utils.fetch_model_blueprint_from_registry(model_constants.MODEL_CURRENT_VERSION)
print("Benchmark model performance")
print("cv_score", benchmark_model["metrics"]["ml"]["cv_score"])
print("validation_regression_score", benchmark_model["metrics"]["ml"]["validation_regression_score"].mean())
print("validation_classification_score", benchmark_model["metrics"]["ml"]["validation_classification_score"].mean())

# New model scores:
print("New model performance")
print("cv_score", chosen_model_blueprint["metrics"]["ml"]["cv_score"])
print("validation_regression_score", chosen_model_blueprint["metrics"]["ml"]["validation_regression_score"].mean())
print("validation_classification_score", chosen_model_blueprint["metrics"]["ml"]["validation_classification_score"].mean())


model_utils.put_model_blueprint_on_registry(chosen_model_blueprint)


