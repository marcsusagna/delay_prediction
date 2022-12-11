import pandas as pd


def compute_delay_by_year(df):
    out_df = (
        df
        .groupby(["scaling_factor", "departure_year"], as_index=False)
        .agg(
            average_delay_time=("delay", "mean"),
            median_delay_time=("delay", "median"),
            max_delay_time=("delay", "max"),
            num_legs=("delay", "count"),
            perc_legs_delayed=("is_delayed", "mean"),
            perc_legs_delayed_more_than_15_min=("is_delayed_more_than_15_min", "mean")
        )
    )
    return out_df


def compute_delay_by_month(df):
    out_df = (
        df
        .groupby(["scaling_factor", "departure_year", "departure_month"], as_index=False)
        .agg(
            legs_with_delay=("is_delayed", "mean"),
            num_legs=("is_delayed", "count")
        )
    )
    return out_df


def compute_delay_by_airport(df):
    out_df = (
        df
        .groupby(["scaling_factor", "departure_year", "dep_ap_scd"], as_index=False)
        .agg(
            legs_with_delay=("is_delayed", "mean"),
            num_legs=("is_delayed", "count")
        )
        .sort_values("num_legs", ascending=False)
    )
    return out_df