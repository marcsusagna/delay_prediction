# To be run from project root directory
import pickle
import os
import sys

import pandas as pd
from sklearn.base import clone

from src.data_preparation import constants as prep_constants

from src.model import utils as model_utils
from src.model import model_evaluation

model_version = sys.argv[1]
model_blueprint = model_utils.fetch_model_blueprint_from_registry(model_version)

df_train = pd.read_parquet(prep_constants.CLEAN_PATH+"train.parquet")
df_validation = pd.read_parquet(prep_constants.CLEAN_PATH+"validation.parquet")
df_all_train = pd.concat([df_train, df_validation])
df_test = pd.read_parquet(prep_constants.CLEAN_PATH+"test.parquet")

## Train on 2021 and predict on 2022 to get estimate of test performance when prediting leg delays
model_to_evaluate = clone(model_blueprint["model"]["untrained_model"])
model_pipeline_wrapper = model_blueprint["model"]["pipeline_wrapper"]

test_score = model_utils.train_and_test_model(model_to_evaluate, model_pipeline_wrapper, df_all_train, df_test)

## Evaluate model on contrafactual test set between 2021 and 2022

# Obtain contrafactual test dataset
pax_increase_param = [model_evaluation.obtain_demand_increase(df_2021=df_all_train, df_2022=df_test)]
df_2022_with_2021_schedule = model_evaluation.match_schedules(df_2021=df_all_train, df_2022=df_test)
df_2022_with_2021_schedule["total_pax_2022"] = df_2022_with_2021_schedule["total_pax"]
df_contrafactual_test = model_evaluation.obtain_contrafactual_dataset(
    df_2022_with_2021_schedule,
    pax_increase_param,
    "total_pax_2021"
)

# Predict on contrafactual dataset
X_contrafactual_test, y_contrafactual_test = (
    model_utils
    .create_X_and_y(df_contrafactual_test, model_pipeline_wrapper, "binary")
)

# Real delays in flights from 2021 schedule that replicated in 2022 schedule:
delays_2022_for_2021_schedule = y_contrafactual_test.mean()

# Predicted delays according to contrafactual analysis with the given pax increase:
predicted_delays_2022 = model_to_evaluate.predict(X_contrafactual_test).mean()

# Update template with scores:
model_blueprint["metrics"]["test_score"] = test_score
model_blueprint["metrics"]["delays_2022_for_2021_schedule"] = delays_2022_for_2021_schedule
model_blueprint["metrics"]["predicted_delays_2022"] = predicted_delays_2022

model_utils.put_model_blueprint_on_registry(model_blueprint)