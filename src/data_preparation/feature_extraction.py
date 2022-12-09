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
        return out_df

    def date_features(self, input_df):
        out_df = (
            input_df
            .assign(
                departure_month=input_df["dep_day_scd"].dt.month,
                departure_year=input_df["dep_day_scd"].dt.year
            )
        )
        return out_df