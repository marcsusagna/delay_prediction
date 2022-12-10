import pickle
from joblib import dump, load
import os
from sklearn.metrics import accuracy_score

from sklearn.base import clone


def create_X_and_y(df, pipeline_wrapper, response_variable_type):
    """
    :param df: dataframe with features and response variable
    :param pipeline_wrapper: metadata on df (which variables are what: response, numeric features, cat features...)
    :param response_variable_type: Either binary or numeric depending on the task at hand
    :return: X and y matrix and vector
    """
    X = df.loc[:, pipeline_wrapper.all_features]
    y = df.loc[:, pipeline_wrapper.response_variables[response_variable_type]]
    return X, y


def train_model(model, pipeline_wrapper, df_train):
    X_train, y_train = create_X_and_y(df_train, pipeline_wrapper, "binary")
    model.fit(X_train, y_train)


def obtain_test_score(trained_model, pipeline_wrapper, df_test):
    X_test, y_test = create_X_and_y(df_test, pipeline_wrapper, "binary")
    val_score = accuracy_score(y_test, trained_model.predict(X_test))
    return val_score

def train_and_test_model(model, pipeline_wrapper, df_train, df_test):
    train_model(model, pipeline_wrapper, df_train)
    test_score = obtain_test_score(model, pipeline_wrapper, df_test)
    return test_score

def create_model_blueprint(model_version_tag, untrained_chosen_model, pipeline_wrapper, cv_scores, df_train,
                           df_validation):
    chosen_model = clone(untrained_chosen_model)
    cv_scores_chosen_model = cv_scores

    # Train model to get validation scores:
    val_score = train_and_test_model(chosen_model, pipeline_wrapper, df_train, df_validation)
    model_blueprint = {
        "model": {
            "model_version": model_version_tag,
            "pipeline_wrapper": pipeline_wrapper,
            "untrained_model": untrained_chosen_model
        },
        "metrics": {
            "cv_scores": cv_scores_chosen_model,
            "validation_score": val_score
        }
    }
    return model_blueprint


def put_model_blueprint_on_registry(model_metadata):
    model_path = "model_registry/" + model_metadata["model"]["model_version"] + "/"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model_metadata, open(model_path + "blueprint.pkl", 'wb'))


def fetch_model_blueprint_from_registry(model_version_tag):
    model_path = "model_registry/" + model_version_tag + "/"
    return pickle.load(open(model_path + "blueprint.pkl", 'rb'))


def store_trained_model(trained_model, model_version_tag):
    model_path = "model_registry/" + model_version_tag + "/"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(trained_model, model_path + "trained_model.joblib")


def fetch_trained_model(model_version_tag):
    model_path = "model_registry/" + model_version_tag + "/"
    model = load(model_path + "trained_model.joblib")
    return model