# This script runs in a container on PROD environment

import sys
import os
import pickle
from joblib import load

import pandas as pd

from src.model import model_visualization

OUTPUT_PATH = "model_output/"
# Import trained model and its blueprint
model_path = "model_registry/v0.0.1/"
model = load(model_path+"trained_model.joblib")
model_blueprint = pickle.load(open(model_path + "blueprint.pkl", 'rb'))
# Import 2022 data: Base for the contrafactual test
df_test = pd.read_parquet("data/clean/test.parquet")

# Prepare contrafactual test:
all_scaling_factors = [float(x) for x in sys.argv[1:]]

df_all_contrafactual_test = (
    pd
    .concat(
        [df_test.copy().assign(scaling_factor=1+x, original_total_pax=df_test["total_pax"], total_pax=df_test["total_pax"]*(1+x)) for x in all_scaling_factors]
    )
)

X_train = df_all_contrafactual_test.loc[:, model_blueprint["model"]["pipe_wrapper"].id_col + model_blueprint["model"]["pipe_wrapper"].all_features]

# Predict on contrafactual test:
df_all_contrafactual_test["is_delayed"] = model.predict(X_train)

# Obtain metrics on the contrafactual test:
# Metric 1: Actual passenger increase:
actual_pax_increase = (
    df_all_contrafactual_test
    .groupby("scaling_factor", as_index=False)
    .agg(
        actual_total_pax_after_scaling=("total_pax", "sum"),
        original_total_pax=("original_total_pax", "sum")
    )
)

actual_pax_increase = (
    actual_pax_increase
    .assign(
        absorbed_increase=(
            (actual_pax_increase["actual_total_pax_after_scaling"]
            - actual_pax_increase["original_total_pax"])
            /actual_pax_increase["original_total_pax"]
        )
    )
)

# Obtain metrics for model visualization

df_train = pd.read_parquet("data/clean/train.parquet")
df_validation = pd.read_parquet("data/clean/validation.parquet")
df_test = pd.read_parquet("data/clean/test.parquet")


df_all_dates = pd.concat([df_train, df_validation, df_test])
df_all_dates["scaling_factor"] = 1
df_all_contrafactual_test["departure_year"] = 2023
df_all_dates = pd.concat([df_all_dates, df_all_contrafactual_test])

# Metric 2: expected delays per year
delay_by_year = model_visualization.compute_delay_by_year(df_all_dates)

# Metric 3: delays per month
delay_by_month = model_visualization.compute_delay_by_month(df_all_dates)

# Metric 4: delays per departure airport
delay_by_dep_ap = model_visualization.compute_delay_by_airport(df_all_dates)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
delay_by_year.to_parquet(OUTPUT_PATH+"delay_by_year.parquet", index=False)
delay_by_month.to_parquet(OUTPUT_PATH+"delay_by_month.parquet", index=False)
delay_by_dep_ap.to_parquet(OUTPUT_PATH+"delay_by_dep_ap.parquet", index=False)

# Temporarily, until API developed:

print(delay_by_year)