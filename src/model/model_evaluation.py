import pandas as pd
import numpy as np

def obtain_demand_increase(df_2021, df_2022):
    pax_2021 = df_2021["total_pax"].sum()
    pax_2022 = df_2022["total_pax"].sum()
    pax_increase = (pax_2022 - pax_2021) / pax_2021
    return pax_increase


def match_schedules(df_2021, df_2022):
    """
    Restrict the flights on df_2022 to only those that were already scheduled in 2021
    At inference time, we do the same between 2022 and 2023, with the difference that we still don't know what
    happened in 2023

    Obtain the amount of passengers per flight in a year. The flight is identified by carrier, flight no and dep date.
    Then
    :param df_2021:
    :return:
    """
    # We compute max because a flight has many legs and some pax repeat over them, so we would be double counting
    # if we were to sum
    schedule_2021 = (
        df_2021
        .groupby("flight_id", as_index=False)
        .agg(total_pax_2021=("total_pax", "max"))
    )

    # Obtain flights in 2022 that have the schedule from 2021
    df_2022_with_2021_schedule = (
        df_2022
        .merge(
            schedule_2021,
            on="flight_id",
            how="inner"
        )
    )
    return df_2022_with_2021_schedule


def obtain_contrafactual_dataset(df_test, relative_increases: list, column_prev_year_pax):
    """

    :param df_test: base_df with all legs that happened in a previous schedule and are supposed to repeat in the testing period
    :param relative_increases: list of values between [0-1] representing increased customer demand
    :return:
    """
    df_all_contrafactual_test = (
        pd
        .concat(
            [
                (
                    df_test.copy()
                    .assign(
                        scaling_factor=1 + x,
                        original_total_pax=lambda df: df[column_prev_year_pax],
                        total_pax_uncapped=lambda df: df[column_prev_year_pax] * (1 + x),
                        free_pax_seats=lambda df: df["pax_seats"]-df["dhc_fln"]-df["xcr_fln"],
                        total_pax=lambda df: df.loc[:, ["total_pax_uncapped", "free_pax_seats"]].min(axis=1),
                        ac_occupancy=lambda df: df["total_pax"]/df["free_pax_seats"]
                    )
                )
                for x in relative_increases
            ]
        )
    )
    return df_all_contrafactual_test


def characterize_delay_distribution(delay_times):
    out_dict = {
        # Distribution of time delayed
        "average_delay_time": np.mean(delay_times),
        "median_delay_time": np.median(delay_times),
        "quantiles_delay_time": {x: np.quantile(delay_times, x) for x in [0.95, 0.99, 0.999]},
        "max_delay_time": np.max(delay_times),
        #  Distribution of num legs delayed
        "num_legs": len(delay_times),
        "perc_legs_delayed": np.mean((delay_times > 0)),
        "perc_legs_delayed_more_than_15_min": np.mean((delay_times > 15))
    }
    return out_dict


def obtain_business_metrics(df, pax_column, delay_time_column):
    """
    :param df: must contain a response variable with delay times and total number of pax
    :return:
    """
    affected_pax = df[df[delay_time_column] > 0][pax_column].sum()
    total_lost_time = (df[df[delay_time_column] > 0][delay_time_column].sum())/(60*24)
    return affected_pax, total_lost_time