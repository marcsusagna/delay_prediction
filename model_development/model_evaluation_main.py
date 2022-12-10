# To be run from project root directory

import pickle
import os

import pandas as pd

from sklearn.base import clone

from sklearn.metrics import accuracy_score

model_path = "model_registry/v0.0.1/"

# Onboard model blueprint
model_blueprint = pickle.load(open(model_path + "blueprint.pkl", 'rb'))
chosen_model = model_blueprint["model"]
# Onboard train + validation data

df_train = pd.read_parquet("data/clean/train.parquet")
df_validation = pd.read_parquet("data/clean/validation.parquet")
df_all_train = pd.concat([df_train, df_validation])

X_train = df_all_train.loc[:, chosen_model["pipe_wrapper"].id_col + chosen_model["pipe_wrapper"].all_features]
y_train = df_all_train.loc[:, chosen_model["pipe_wrapper"].response_variables[:1]]

# Train model
trained_model = clone(chosen_model["untrained_model"])
trained_model.fit(X_train, y_train)

# Evaluate model on test set to get delay prediction performance:
df_test = pd.read_parquet("data/clean/test.parquet")
X_test = df_test.loc[:, chosen_model["pipe_wrapper"].id_col + chosen_model["pipe_wrapper"].all_features]
y_test = df_test.loc[:, chosen_model["pipe_wrapper"].response_variables[:1]]

test_score = accuracy_score(y_test, trained_model.predict(X_test))

# Evaluate model on contrafactual test set between 2021 and 2022

# Obtain pax increase between 2021 and 2022
pax_2021 = df_all_train.total_pax.sum()
pax_2022 = df_test.total_pax.sum()
pax_increase_param = (pax_2022-pax_2021)/pax_2021

# Obtain number of passangers per flight in 2021
schedule_2021 = df_all_train.groupby("flight_id", as_index=False).agg({"total_pax": "max"}).rename(columns={"total_pax": "total_pax_2021"})

# Obtain contrafactual test set: Same schedule as 2021
df_contrafactual_test = (
    df_test
    .merge(
        schedule_2021,
        on="flight_id",
        how="inner"
    )
)
df_contrafactual_test["total_pax_forecast"] = df_contrafactual_test["total_pax_2021"]*(1+pax_increase_param)

# Predict for contrafactual test set:
df_contrafactual_test["total_pax"] = df_contrafactual_test["total_pax_forecast"]

X_contrafactual_test = df_contrafactual_test.loc[:, chosen_model["pipe_wrapper"].id_col + chosen_model["pipe_wrapper"].all_features]
y_contrafactual_test = df_contrafactual_test.loc[:, chosen_model["pipe_wrapper"].response_variables[:1]]

#contrafactual_test_score = accuracy_score(y_contrafactual_test, trained_model.predict(X_contrafactual_test))

# Real delays in flights from 2021 schedule that replicated in 2022 schedule:
delays_2022_for_2021_schedule = y_contrafactual_test.mean()
delays_2022 = y_test.mean()

# Predicted delays according to contrafactual analysis with the given pax increase:
predicted_delays_2022 = trained_model.predict(X_contrafactual_test).mean()

# Update template with scores:
model_blueprint["metrics"]["test_score"] = test_score
model_blueprint["metrics"]["delays_2022_for_2021_schedule"] = delays_2022_for_2021_schedule
model_blueprint["metrics"]["predicted_delays_2022"] = predicted_delays_2022

# Update model registry with new metrics:
model_blueprint_path = "model_registry/v0.0.1/blueprint.pkl"
os.makedirs(os.path.dirname(model_blueprint_path), exist_ok=True)
pickle.dump(model_blueprint, open(model_blueprint_path, 'wb'))