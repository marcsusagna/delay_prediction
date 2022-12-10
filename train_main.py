import pickle
from joblib import dump
import os

import pandas as pd

from sklearn.base import clone

model_path = "model_registry/v0.0.1/"
model_blueprint = pickle.load(open(model_path + "blueprint.pkl", 'rb'))
chosen_model = model_blueprint["model"]

# Create training set with all data
df_train = pd.read_parquet("data/clean/train.parquet")
df_validation = pd.read_parquet("data/clean/validation.parquet")
df_test = pd.read_parquet("data/clean/test.parquet")

df_all_train = pd.concat([df_train, df_validation, df_test])

X_train = df_all_train.loc[:, chosen_model["pipe_wrapper"].id_col + chosen_model["pipe_wrapper"].all_features]
y_train = df_all_train.loc[:, chosen_model["pipe_wrapper"].response_variables[:1]]

# Train the model
trained_model = clone(chosen_model["untrained_model"])
trained_model.fit(X_train, y_train)

# Store trained model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
dump(trained_model, model_path+"trained_model.joblib")