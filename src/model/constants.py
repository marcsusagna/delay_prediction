MODEL_CURRENT_VERSION = "v0.0.1"
MODEL_NEW_VERSION = "v0.0.2"

# variables
ID_COL = [
    "leg_i"
]

RESPONSE_VARIABLES = {
    "binary": "is_delayed",
    "numeric": "total_delay_time"
}

NUMERIC_VARIABLES = [
    "total_pax",
    "ac_occupancy"
]

CATEGORICAL_VARIABLES = [
    "dep_ap_scd"
]