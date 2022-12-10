import pickle
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


def train_model(model, pipeline_wrapper, base_df):
    X_train, y_train = create_X_and_y(base_df, pipeline_wrapper, "binary")
    model.fit(X_train, y_train)


def obtain_validation_score(trained_model, pipeline_wrapper, validation_df):
    X_validation, y_validation = create_X_and_y(validation_df, pipeline_wrapper, "binary")
    val_score = accuracy_score(y_validation, trained_model.predict(X_validation))
    return val_score


def create_model_blueprint(model_version_tag, untrained_chosen_model, pipeline_wrapper, cv_scores, df_train,
                           df_validation):
    chosen_model = clone(untrained_chosen_model)
    cv_scores_chosen_model = cv_scores

    # Train model to get validation scores:
    train_model(chosen_model, pipeline_wrapper, df_train)
    val_score = obtain_validation_score(chosen_model, pipeline_wrapper, df_validation)
    model_blueprint = {
        "model": {
            "model_version": model_version_tag,
            "pipe_wrapper": pipeline_wrapper,
            "untrained_model": untrained_chosen_model
        },
        "metrics": {
            "cv_scores": cv_scores_chosen_model,
            "validation_score": val_score
        }
    }
    return model_blueprint


def put_model_on_registry(model_metadata):
    model_path = "model_registry/" + model_metadata["model"]["model_version"] + "/"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model_metadata, open(model_path + "blueprint.pkl", 'wb'))