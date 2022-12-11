import os

from src.data_preparation import data_onboarding
from src.data_preparation import constants

delay_onboarder = data_onboarding.DelayOnboarding(constants.SOURCE_PATH+"delay.csv", constants.DELAY_COLUMNS_TO_ONBOARD)
df_delay = delay_onboarder.prepare_dataset(constants.DELAY_OUTPUT_COLUMNS_MAP)

fis_onboarder = data_onboarding.FisOnboarding(constants.SOURCE_PATH+"fis.csv", constants.FIS_COLUMNS_TO_ONBOARD)
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
    .merge(
        df_delay.rename(columns={"total_delay_time": "prev_leg_total_delay_time"}),
        left_on="flt_leg_i_prev",
        right_on="leg_i",
        how="left"
    )
)

# Obtain airport metrics:
output_df["is_cancelled"] = (output_df["leg_state"] == "CNL").astype(int)
cancellations_per_ap = (
    output_df
    .groupby(["dep_ap_scd", "dep_day_scd"], as_index=False)
    .agg(
        num_cancelled_flights_in_ap=("is_cancelled", "sum"),
        num_scd_flights_in_ap=("is_cancelled", "count")
    )
)
output_df = (
    output_df
    .merge(
        cancellations_per_ap,
        on=["dep_ap_scd", "dep_day_scd"],
        how="inner"
    )
)
# Drop cancelled flights
output_df = output_df[output_df["leg_state"] != "CNL"]

# Clean up some variables wrongly parsed:
output_df["season"] = output_df["season"].str.replace(" ", "")

# Onboard aircraft capacity (pax seats):
ac_capacity = data_onboarding.AircraftCapacityCreator()
df_ac_capacity = ac_capacity.return_aircraft_capacity()

output_df = (
    output_df
    .merge(
        df_ac_capacity,
        on="ac_subtype",
        how="inner"
    )
)

# Write to parquet to reduce size (better compression) and keep schema in file metadata
os.makedirs(os.path.dirname(constants.ONBOARDED_PATH), exist_ok=True)
output_df.to_parquet(constants.ONBOARDED_PATH+"legs_with_delay_info.parquet", index=False)
