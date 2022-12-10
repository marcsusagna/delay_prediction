# This script runs in a container on PROD environment
import sys

import pandas as pd
from sklearn.base import clone

from src.data_preparation import constants as prep_constants

from src.model import utils as model_utils

model_version = sys.argv[1]
model_blueprint = model_utils.fetch_model_blueprint_from_registry(model_version)

df_train = pd.read_parquet(prep_constants.CLEAN_PATH+"train.parquet")
df_validation = pd.read_parquet(prep_constants.CLEAN_PATH+"validation.parquet")
df_test = pd.read_parquet(prep_constants.CLEAN_PATH+"test.parquet")
df_all_train = pd.concat([df_train, df_validation, df_test])

final_model = clone(model_blueprint["model"]["untrained_model"])
model_pipeline_wrapper = model_blueprint["model"]["pipeline_wrapper"]
model_utils.train_model(
    model=final_model,
    pipeline_wrapper=model_pipeline_wrapper,
    df_train=df_all_train
)

model_utils.store_trained_model(final_model, model_version)