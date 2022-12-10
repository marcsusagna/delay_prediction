import pandas as pd

class FeatureExtractor():
    """
    Class to wrap all features extracted by group
    """
    def __init__(self, df):
        self.df = df

    def passanger_features(self, input_df):
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
                ac_occupancy=(
                    out_df.total_pax/out_df.pax_seats
                )
            )
        )
        return out_df

    def date_features(self, input_df):
        out_df = (
            input_df
            .assign(
                departure_day=input_df["dep_day_scd"].dt.day,
                departure_month=input_df["dep_day_scd"].dt.month,
                departure_year=input_df["dep_day_scd"].dt.year
            )
        )
        return out_df

    def id_features(self, input_df):
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