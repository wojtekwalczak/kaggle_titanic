"""Represents the best (based on grid search) settings for
LogisticRegression."""

from sklearn.linear_model import LogisticRegression

class LogisticRegressionClf(object):
    def __init__(self):
        self.clf = LogisticRegression()

    def get_classifier(self):
        return self.clf
