import os

from src.data_preparation import data_onboarding
from src.data_preparation import constants

delay_onboarder = data_onboarding.DelayOnboarding(constants.DELAY_DATASET_PATH, constants.DELAY_COLUMNS_TO_ONBOARD)
df_delay = delay_onboarder.prepare_dataset(constants.DELAY_OUTPUT_COLUMNS_MAP)

fis_onboarder = data_onboarding.FisOnboarding(constants.FIS_DATASET_PATH, constants.FIS_COLUMNS_TO_ONBOARD)
df_fis = fis_onboarder.prepare_dataset(constants.FIS_DATE_COLS)

# There are three flights for which there is no record in the delay table, we drop them (inner)
# Most likely they were cancelled. They have 0 passangers
output_df = (
    df_fis
    .merge(
        df_delay,
        on="leg_i",
        how="inner"
    )
)
# Write to parquet to reduce size (better compression) and keep schema in file metadata
ONBOARDED_PATH = "data/onboarded/"
os.makedirs(os.path.dirname(ONBOARDED_PATH), exist_ok=True)
output_df.to_parquet(ONBOARDED_PATH+"flights_with_delay.parquet", index=False)
