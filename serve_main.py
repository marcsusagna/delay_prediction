# This script runs in a container on PROD environment
# run as python serve_main.py model_tag inc1 inc2 inc3
# where inc1, inc2, inc3... are all the relative increases in customer demand that you want to test
import sys
import os
import pandas as pd

from src.data_preparation import constants as prep_constants

from src.model import utils as model_utils
from src.model import model_evaluation
from src.model import model_visualization

model_version = sys.argv[1]
model = model_utils.fetch_trained_model(model_version)
model_blueprint = model_utils.fetch_model_blueprint_from_registry(model_version)

df_test = pd.read_parquet(prep_constants.CLEAN_PATH+"test.parquet")

all_rel_increase = [float(x) for x in sys.argv[2:]]
df_all_contrafactual_test = model_evaluation.obtain_contrafactual_dataset(
    df_test=df_test,
    relative_increases=all_rel_increase,
    column_prev_year_pax="total_pax"
)

X_contrafactual_test, _ = (
    model_utils
    .create_X_and_y(df_all_contrafactual_test, model_blueprint["model"]["pipeline_wrapper"], "binary")
)

df_all_contrafactual_test["is_delayed"] = model.predict(X_contrafactual_test)

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

print("Actual pax increase")
print(actual_pax_increase)

# Obtain metrics for model visualization

df_train = pd.read_parquet("data/clean/train.parquet")
df_validation = pd.read_parquet("data/clean/validation.parquet")

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

output_path = "model_output/"+model_version+"/"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
delay_by_year.to_parquet(output_path+"delay_by_year.parquet", index=False)
delay_by_month.to_parquet(output_path+"delay_by_month.parquet", index=False)
delay_by_dep_ap.to_parquet(output_path+"delay_by_dep_ap.parquet", index=False)

# Temporarily, until API developed:

print(delay_by_year)