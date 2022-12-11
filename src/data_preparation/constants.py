# Data folder structure
SOURCE_PATH = "data/source/"
ONBOARDED_PATH = "data/onboarded/"
CLEAN_PATH = "data/clean/"

# Onboarding constants
DELAY_COLUMNS_TO_ONBOARD = [
    "wh_fdel_leg_i",
    "wh_fdel_delay_time"
]

DELAY_OUTPUT_COLUMNS_MAP = {
    "wh_fdel_leg_i": "leg_i",
    "wh_fdel_delay_time": "total_delay_time"
}

# Use UTC columns
FIS_COLUMNS_TO_ONBOARD = [
    "wh_fleg_leg_i",
    "wh_fleg_flt_carrier",
    "wh_fleg_ac_subtype",
    "wh_fleg_ac_owner",
    "wh_fleg_flt_no",
    "wh_fleg_dep_ap_scd",
    "wh_fleg_arr_ap_scd",
    "wh_fleg_arr_ap_act",
    "wh_fleg_dep_day_scd",
    "wh_fleg_season",
    "wh_fleg_diverted",
    "wh_fleg_leg_state",
    "wh_fleg_flt_leg_i_prev",
    "wh_fleg_rot_leg_i_prev",
    "wh_fleg_pax_fln_c",
    "wh_fleg_pax_fln_y",
    "wh_fleg_pax_fln_f",
    "wh_fleg_pax_fln_e",
    "wh_fleg_dhc_fln",
    "wh_fleg_xcr_fln",
    "wh_fleg_baggage_weight",
    "wh_fleg_baggage_pieces",
    "wh_fleg_cargo",
    "wh_fleg_mail",
    "wh_fleg_leg_dist_scd",
    "wh_fleg_leg_dist_act",
    "wh_fleg_flt_dist_scd",
    "wh_fleg_flt_dist_act",
    "wh_fleg_flt_legs",
    "wh_fleg_offblock_dt",
    "wh_fleg_airborne_dt",
    "wh_fleg_landing_dt",
    "wh_fleg_onblock_dt",
    "wh_fleg_entry_dt",
    "wh_fleg_dep_dt_scd",
    "wh_fleg_dep_dt_scd_loc",
    "wh_fleg_arr_dt_scd_loc"
]

FIS_DATE_COLS = {
    "wh_fleg_dep_day_scd": True,
    "wh_fleg_offblock_dt": False,
    "wh_fleg_airborne_dt": False,
    "wh_fleg_landing_dt": False,
    "wh_fleg_onblock_dt": False,
    "wh_fleg_entry_dt": False,
    "wh_fleg_dep_dt_scd": False,
    "wh_fleg_dep_dt_scd_loc": False,
    "wh_fleg_arr_dt_scd_loc": False,
}

# Data preparation constants
TRAIN_VAL_SPLIT_SEED = 4891
VAL_FRACTION = 0.2
