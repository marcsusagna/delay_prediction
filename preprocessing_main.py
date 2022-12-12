import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preparation import feature_extraction
from src.data_preparation import constants


base_df = pd.read_parquet(constants.ONBOARDED_PATH+"legs_with_delay_info.parquet")

# Feature Extraction
df_with_features = feature_extraction.passenger_features(base_df)
df_with_features = feature_extraction.categorical_features(df_with_features)
df_with_features = feature_extraction.date_features(df_with_features)
df_with_features = feature_extraction.time_difference_features(df_with_features)
df_with_features = feature_extraction.numeric_features(df_with_features)
df_with_features = feature_extraction.id_features(df_with_features)

# Structural NaN imputation:
# column rel_leg_distance: 0/0 conceptually means all the distance of the flight done by this leg: impute 1
df_with_features["rel_leg_distance"] = df_with_features["rel_leg_distance"].fillna(1)
# column prev_leg_total_delay_time: Mostly due to single legged flights. Impute 0, we assume no delay accumulated
df_with_features["prev_leg_total_delay_time"] = df_with_features["prev_leg_total_delay_time"].fillna(0)

# Data split: Splitting in three sets:
# - Train: 80% of 2021. Use: Model development and selection through CV within it
# - Validation: 20% of 2021: Use: Assess how well we are doing just predicting delays
# - Test 100% of 2022: Use: We'll assess how well did we predict delays in 2022 assuming a pax increase w.r.t 2021

df_2021 = df_with_features[df_with_features.departure_year == 2021]
df_test = df_with_features[df_with_features.departure_year == 2022]
df_train, df_validation = train_test_split(
    df_2021,
    test_size=constants.VAL_FRACTION,
    random_state=constants.TRAIN_VAL_SPLIT_SEED
)

os.makedirs(os.path.dirname(constants.CLEAN_PATH), exist_ok=True)

df_test.to_parquet(constants.CLEAN_PATH+"test.parquet", index=False)
df_train.to_parquet(constants.CLEAN_PATH+"train.parquet", index=False)
df_validation.to_parquet(constants.CLEAN_PATH+"validation.parquet", index=False)