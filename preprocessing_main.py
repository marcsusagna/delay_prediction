from src.data_preparation.feature_extraction import FeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split

base_df = pd.read_parquet("data/onboarded/flights_with_delay.parquet")

# Feature Extraction
feature_extractor = FeatureExtractor(base_df)
df_with_features = feature_extractor.passanger_features(base_df)
df_with_features = feature_extractor.date_features(df_with_features)

# Data split: Splitting in three sets:
# - Train: 80% of 2021. Use: Model development and selection through CV within it
# - Validation: 20% of 2021: Use: Assess how well we are doing just predicting delays
# - Test 100% of 2022: Use: We'll assess how well did we predict delays in 2022 assuming a pax increase w.r.t 2021

df_2021 = df_with_features[df_with_features.departure_year == 2021]
df_test = df_with_features[df_with_features.departure_year == 2022]
df_train, df_validation = train_test_split(df_2021, test_size=0.2)

df_test.to_parquet("data/clean/test.parquet", index=False)
df_train.to_parquet("data/clean/train.parquet", index=False)
df_validation.to_parquet("data/clean/validation.parquet", index=False)