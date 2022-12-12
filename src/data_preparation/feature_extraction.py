import pandas as pd
import numpy as np


def passenger_features(input_df):
    out_df = (
        input_df
        .assign(
            total_pax=(
                input_df.pax_fln_c
                + input_df.pax_fln_y
                + input_df.pax_fln_f
                + input_df.pax_fln_e
            )
        )
    )
    out_df = (
        out_df
        .assign(
            # refers to aircraft occupancy over free seats (not used by crew)
            ac_occupancy=(
                out_df.total_pax/(out_df.pax_seats - out_df.dhc_fln - out_df.xcr_fln)
            )
        )
    )
    return out_df


def categorical_features(input_df):
    out_df = (
        input_df
        .assign(
            circular_flight= (input_df.arr_ap_scd == input_df.dep_ap_scd).astype(int),
            ap_scd_changed=(input_df.arr_ap_act != input_df.arr_ap_scd).astype(int),
            has_rot_leg_prev=input_df.rot_leg_i_prev.isna().astype(int),
        )
    )
    return out_df


def date_features(input_df):
    out_df = (
        input_df
        .assign(
            # Based on UTC:
            departure_day=input_df["dep_day_scd"].dt.day,
            departure_month=input_df["dep_day_scd"].dt.month,
            departure_year=input_df["dep_day_scd"].dt.year,
        )
    )
    out_df = extract_loc_dependent_features(out_df, "dep_dt_scd_loc")
    out_df = extract_loc_dependent_features(out_df, "arr_dt_scd_loc")
    return out_df


def time_difference_features(input_df):
    out_df = input_df.copy()
    out_df["offblock_to_airborne"] = (input_df["airborne_dt"] - input_df["offblock_dt"]).dt.seconds / 60
    out_df["airborne_to_landing"] = (input_df["landing_dt"] - input_df["airborne_dt"]).dt.seconds / 60
    out_df["landing_to_onblock"] = (input_df["onblock_dt"] - input_df["landing_dt"]).dt.seconds / 60
    out_df["onblock_to_entry"] = (input_df["entry_dt"] - input_df["onblock_dt"]).dt.seconds / 60
    return out_df


def numeric_features(input_df):
    out_df = input_df.copy()
    out_df["rel_leg_distance"] = input_df["leg_dist_scd"]/input_df["flt_dist_scd"]
    return out_df


def id_features(input_df):
    out_df = (
        input_df
        .assign(
            flight_id=(
                    input_df["flt_carrier"]
                    + input_df["flt_no"].astype(str)
                    + "-"
                    + input_df["departure_day"].astype(str)
                    + "/"
                    + input_df["departure_month"].astype(str)
            )
        )
    )
    return out_df


def give_circular_variable(base_column, column_cycle):
    sin_component = np.sin(2 * np.pi * base_column / column_cycle)
    cos_component = np.cos(2 * np.pi * base_column / column_cycle)
    return sin_component, cos_component


def extract_loc_dependent_features(input_df, col_name):
    """
    These features have meaning when computed on the localized timestamp
    :param input_df:
    :param col_name:
    :return:
    """
    col_values = input_df[col_name]

    day_of_month = col_values.dt.day
    day_of_month_sin, day_of_month_cos = give_circular_variable(day_of_month, 30)

    hour_of_day = col_values.dt.hour + col_values.dt.minute / 60
    hour_of_day_sin, hour_of_day_cos = give_circular_variable(hour_of_day, 24)
    part_of_day = pd.cut(hour_of_day, 4, labels=np.arange(4)).astype(str)

    day_of_week = col_values.dt.day_of_week.astype(str)

    out_df = input_df.copy()
    out_df["day_of_month_{}".format(col_name)] = day_of_month
    out_df["day_of_month_sin_{}".format(col_name)] = day_of_month_sin
    out_df["day_of_month_cos_{}".format(col_name)] = day_of_month_cos
    out_df["hour_of_day_{}".format(col_name)] = hour_of_day
    out_df["hour_of_day_sin_{}".format(col_name)] = hour_of_day_sin
    out_df["hour_of_day_cos_{}".format(col_name)] = hour_of_day_cos
    out_df["part_of_day_{}".format(col_name)] = part_of_day
    out_df["day_of_week_{}".format(col_name)] = day_of_week
    return out_df
