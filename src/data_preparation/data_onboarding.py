import pandas as pd


class DatasetOnboarding():
    def __init__(self, dataset_path: str, cols_to_onboard: list):
        self.dataset_path = dataset_path
        self.cols_to_onboard = cols_to_onboard

    def prepare_dataset(self, *args):
        raise NotImplementedError

    def read_dataset(self):
        df = pd.read_csv(self.dataset_path, sep=";")
        return df

    def input_projection(self, df):
        df = df.loc[:, self.cols_to_onboard]
        return df

    def output_projection(self, df, output_columns_map: dict):
        out_df = (
            df
            .rename(
                columns=output_columns_map
            )
        )
        return out_df


class DelayOnboarding(DatasetOnboarding):
    """
    Class to onboard delay dataset

    PK source is [wh_fdel_leg_i, wh_fdel_delay_number], since each flight can have multiple delay reasons
    PK output is [wh_fdel_leg_i]
    """
    def prepare_dataset(self, output_columns_map):
        source_df = self.read_dataset()
        source_df = self.input_projection(source_df)
        agg_df = self.aggregate_delays(source_df)
        out_df = self.output_projection(agg_df, output_columns_map)
        return out_df

    def aggregate_delays(self, df):
        agg_df = df.groupby("wh_fdel_leg_i", as_index=False).agg({"wh_fdel_delay_time": "sum"})
        agg_df = agg_df.assign(is_delayed=(agg_df.wh_fdel_delay_time > 0).astype("int"))
        return agg_df


class FisOnboarding(DatasetOnboarding):
    """
    Class to onboard fis dataset

    PK source is [wh_fleg_leg_i]
    PK output is [wh_fdel_leg_i]
    """
    def prepare_dataset(self, date_columns):
        source_df = self.read_dataset()
        source_df = self.input_projection(source_df)
        self.cast_date_columns(source_df, date_columns)
        # If new columns onboarded, need to implement:
        # - proper type casting (date, numeric wrongly parsed into string
        # - nan coding standardization

        output_columns_map = self.obtain_output_column_names()
        out_df = self.output_projection(source_df, output_columns_map)
        return out_df

    def obtain_output_column_names(self):
        output_columns_map = {x:x[8:] for x in self.cols_to_onboard}
        return output_columns_map

    def cast_date_columns(self, df, date_columns: list):
        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)



class AircraftCapacityCreator():
    # based on aircraft subtype or aircraft registration number and internet search.
    # used max occupancy flight as reference
    # https://www.swiss.com/ch/en/discover/fleet/
    # https://www.airbus.com/en/products-services/defence/military-aircraft/c295
    # Others
    # https://www.airfleets.net/ficheapp/plane-e190-20036.htm using ac_registration
    # https: // www.flyedelweiss.com / EN / fly / flight - information / Pages / fleet.aspx
    # https://www.airfleets.net/ficheapp/plane-e190-20057.htm
    # https://www.seatmaestro.com/airplanes-seat-maps/hello-airlines-airbus-a320-214/
    # https://www.helvetic.com/fleet
    def return_aircraft_capacity(self):
        ac_capacity_data = [
            {"ac_subtype": "223", "pax_seats": 145},
            {"ac_subtype": "320", "pax_seats": 180},
            {"ac_subtype": "221", "pax_seats": 125},
            {"ac_subtype": "290", "pax_seats": 112},
            {"ac_subtype": "32N", "pax_seats": 180},
            {"ac_subtype": "321", "pax_seats": 219},
            {"ac_subtype": "343", "pax_seats": 314},
            {"ac_subtype": "333", "pax_seats": 314},
            {"ac_subtype": "77W", "pax_seats": 340},
            {"ac_subtype": "295", "pax_seats": 134},
            {"ac_subtype": "E90", "pax_seats": 112},
            {"ac_subtype": "32Q", "pax_seats": 219},
            {"ac_subtype": "32B", "pax_seats": 209},
            {"ac_subtype": "77A", "pax_seats": 340},
            {"ac_subtype": "32A", "pax_seats": 174},
            {"ac_subtype": "33W", "pax_seats": 314}
        ]
        return pd.DataFrame(ac_capacity_data)