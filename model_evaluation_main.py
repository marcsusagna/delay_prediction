# To be run from project root directory
import sys

import pandas as pd
import numpy as np
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
print("Test regression score:", test_score[0])
print("Test accuracy score:", test_score[1])
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

contrafactual_test_score = model_utils.train_and_test_model(model_to_evaluate, model_pipeline_wrapper, df_all_train, df_contrafactual_test)
print("Contrafactual test regression score:", contrafactual_test_score[0])
print("Contrafactual test accuracy score:", contrafactual_test_score[1])

# Predict on contrafactual dataset to compare distributions:
X_contrafactual_test, y_contrafactual_test = (
    model_utils
    .create_X_and_y(df_contrafactual_test, model_pipeline_wrapper)
)

# Real delays in flights from 2021 schedule that replicated in 2022 schedule:
delay_2022_for_2021_schedule = model_evaluation.characterize_delay_distribution(y_contrafactual_test.to_numpy())
print("Real delays in 2022 for flights with same schedule as 2021", delay_2022_for_2021_schedule)

business_2022_for_2021_schedule = model_evaluation.obtain_business_metrics(
    df=df_contrafactual_test,
    pax_column="original_total_pax",
    delay_time_column="total_delay_time"
)

# Predicted delays according to contrafactual analysis with the given pax increase:
predicted_delays_2022 = model_to_evaluate.predict(X_contrafactual_test)
df_contrafactual_test["predicted_total_delay_time"] = predicted_delays_2022
predicted_delay_2022_for_2021_schedule = model_evaluation.characterize_delay_distribution(predicted_delays_2022)
print("Predicted delays in 2022 for flights with same schedule as 2021", predicted_delay_2022_for_2021_schedule)

predicted_business_2022_for_2021_schedule = model_evaluation.obtain_business_metrics(
    df=df_contrafactual_test,
    pax_column="total_pax",
    delay_time_column="predicted_total_delay_time"
)

# Update template with scores:
model_blueprint["metrics"]["ml"]["regression_test_score"] = test_score[0]
model_blueprint["metrics"]["ml"]["classification_test_score"] = test_score[1]
model_blueprint["metrics"]["ml"]["regression_contrafactual_test_score"] = contrafactual_test_score[0]
model_blueprint["metrics"]["ml"]["classification_contrafactual_test_score"] = contrafactual_test_score[1]
model_blueprint["metrics"]["delay_distribution"] = {
    "delays_2022_for_2021_schedule" :delay_2022_for_2021_schedule,
    "predicted_delay_2022_for_2021_schedule": predicted_delay_2022_for_2021_schedule
}
model_blueprint["metrics"]["business"] = {
    "business_2022_for_2021_schedule" :business_2022_for_2021_schedule,
    "predicted_business_2022_for_2021_schedule": predicted_business_2022_for_2021_schedule
}

model_utils.put_model_blueprint_on_registry(model_blueprint)


