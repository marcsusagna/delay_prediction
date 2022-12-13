import pickle
from joblib import dump, load
import os
import numpy as np

from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.base import clone


def create_X_and_y(df, pipeline_wrapper):
    """
    :param df: dataframe with features and response variable
    :param pipeline_wrapper: metadata on df (which variables are what: response, numeric features, cat features...)
    :param response_variable_type: Either binary or numeric depending on the task at hand
    :return: X and y matrix and vector
    """
    X = df.loc[:, pipeline_wrapper.all_features]
    y = df.loc[:, pipeline_wrapper.response_variables]
    y_reg = y.iloc[:, 0]
    y_class = (y.iloc[:, 0] > 0).astype(int)
    return X, y_reg, y_class


def train_model(model, pipeline_wrapper, df_train):
    X_train, y_train_reg, y_train_class = create_X_and_y(df_train, pipeline_wrapper)
    model["reg"].fit(X_train, y_train_reg)
    model["class"].fit(X_train, y_train_class)

def predict_dual_model(model, X):
    y_pred_reg = model["reg"].predict(X)
    y_pred_class = model["class"].predict(X)
    return y_pred_reg, y_pred_class

def obtain_test_score(trained_model, pipeline_wrapper, df_test):
    X_test, y_test_reg, y_test_class = create_X_and_y(df_test, pipeline_wrapper)
    y_pred_reg, y_pred_class = predict_dual_model(trained_model, X_test)
    regression_score = r2_score(y_test_reg, y_pred_reg)
    classification_score = accuracy_score(y_test_class, y_pred_class)
    return regression_score, classification_score


def train_and_test_model(model, pipeline_wrapper, df_train, df_test):
    train_model(model, pipeline_wrapper, df_train)
    test_score = obtain_test_score(model, pipeline_wrapper, df_test)
    return test_score


def create_model_blueprint(model_version_tag, untrained_chosen_model, pipeline_wrapper, cv_score, df_train,
                           df_validation):
    chosen_model = clone_model(untrained_chosen_model)
    cv_scores_chosen_model = cv_score

    # Train model to get validation scores:
    val_score = train_and_test_model(chosen_model, pipeline_wrapper, df_train, df_validation)
    model_blueprint = {
        "model": {
            "model_version": model_version_tag,
            "pipeline_wrapper": pipeline_wrapper,
            "untrained_model": untrained_chosen_model
        },
        "metrics": {
            "ml": {
                "cv_score": cv_scores_chosen_model,
                "validation_regression_score": val_score[0],
                "validation_classification_score": val_score[1]
            }
        }
    }
    return model_blueprint


def clone_model(model):
    return {k: clone(v) for k, v in model.items()}



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


def find_best_cv(cv_dict):
    idx_best = np.argmax([v["cv_scores"].mean() for k, v in cv_dict.items()])
    best_key = list(cv_dict.keys())[idx_best]
    print("Chosen model:", best_key)
    return clone(cv_dict[best_key]["model"]), cv_dict[best_key]["cv_scores"]

def add_cv_result(cv_dict, model_name, model, X, y, cv_folds):
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    cv_dict[model_name] = {
        "model": model,
        "cv_score": cv_scores
    }