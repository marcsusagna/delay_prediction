from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import numpy as np

class zero_inflated_estimator(BaseEstimator):

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        non_zero_idx = y.iloc[:, 0]>0
        self.classifier.fit(X, non_zero_idx.astype(int))
        self.regressor.fit(X[non_zero_idx], y[non_zero_idx])
        #return self

    def predict(self, X):

        binary_predict = self.classifier.predict(X)
        reg_predict = self.regressor.predict(X)[:, 0]
        return binary_predict*reg_predict

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)


class zero_inflated_log_estimator(BaseEstimator):

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def stand_y(self, y):
        return (y - self.log_y_non_zero_mean)/self.log_y_non_zero_std

    def fit(self, X, y):
        non_zero_idx = y.iloc[:, 0] > 0
        self.classifier.fit(X, non_zero_idx.astype(int))
        X_non_zero = X[non_zero_idx]
        log_y_non_zero = np.log(y[non_zero_idx])
        self.log_y_non_zero_mean = log_y_non_zero.mean().iloc[0]
        self.log_y_non_zero_std = log_y_non_zero.std().iloc[0]
        log_y_non_zero_stand = self.stand_y(log_y_non_zero)
        self.regressor.fit(X_non_zero, log_y_non_zero_stand)
        #return self

    def predict(self, X):

        binary_predict = self.classifier.predict(X)
        reg_predict = np.exp(self.regressor.predict(X)[:, 0]*self.log_y_non_zero_std + self.log_y_non_zero_mean)
        return binary_predict*reg_predict

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)