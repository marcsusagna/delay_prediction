import pandas as pd


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
            [df_test.copy().assign(scaling_factor=1 + x, original_total_pax=df_test[column_prev_year_pax],
                                   total_pax=df_test[column_prev_year_pax] * (1 + x)) for x in relative_increases]
        )
    )
    return df_all_contrafactual_test
