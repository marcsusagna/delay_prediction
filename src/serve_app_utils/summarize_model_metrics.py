import pandas as pd

from ..model import constants
from ..model import utils as model_utils

def get_metrics_in_lists(my_dict, label):
    metric_names = list(my_dict.keys())
    metric_values = list(my_dict.values())
    metric_family = [label] * len(metric_names)
    return metric_names, metric_values, metric_family


def extract_specific_scenario_delay_metrics(subdict, label):
    tmp_dict = subdict.copy()
    tmp_dict.pop("quantiles_delay_time")
    return get_metrics_in_lists(tmp_dict, label)


def obtain_delay_metrics(dict_business_metrics):
    real_2022 = extract_specific_scenario_delay_metrics(
        dict_business_metrics["delays_2022_for_2021_schedule"],
        "real_2022_in_2021_schedule"
    )
    predicted_2022 = extract_specific_scenario_delay_metrics(
        dict_business_metrics["predicted_delay_2022_for_2021_schedule"],
        "predicted_2022_in_2021_schedule"
    )
    return real_2022, predicted_2022


def summarize_model_metrics():
    model_version = constants.MODEL_CURRENT_VERSION
    model_blueprint = model_utils.fetch_model_blueprint_from_registry(model_version)

    ml_metrics = get_metrics_in_lists(model_blueprint["metrics"]["ml"], "machine_learning")
    delay_metrics = obtain_delay_metrics(model_blueprint["metrics"]["delay_distribution"])

    all_names = ml_metrics[0] + delay_metrics[0][0] + delay_metrics[1][0]
    all_values = ml_metrics[1] + delay_metrics[0][1] + delay_metrics[1][1]
    all_labels = ml_metrics[2] + delay_metrics[0][2] + delay_metrics[1][2]

    my_df = pd.DataFrame({
        "metric_family": all_labels,
        "metric_names": all_names,
        "metric_values": all_values
    })
    return my_df