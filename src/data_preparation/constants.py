DELAY_COLUMNS_TO_ONBOARD = [
    "wh_fdel_leg_i",
    "wh_fdel_delay_time"
]

DELAY_OUTPUT_COLUMNS_MAP = {
    "wh_fdel_leg_i": "leg_i",
    "wh_fdel_delay_time": "total_delay_time",
    "is_delayed": "is_delayed"
}

DELAY_DATASET_PATH = "data/source/delay.csv"

# Use UTC columns
FIS_COLUMNS_TO_ONBOARD = [
    "wh_fleg_leg_i",
    "wh_fleg_pax_fln_c",
    "wh_fleg_pax_fln_y",
    "wh_fleg_pax_fln_f",
    "wh_fleg_pax_fln_e",
    "wh_fleg_dep_ap_scd",
    "wh_fleg_dep_day_scd"
]

FIS_DATE_COLS = ["wh_fleg_dep_day_scd"]

FIS_DATASET_PATH = "data/source/fis.csv"