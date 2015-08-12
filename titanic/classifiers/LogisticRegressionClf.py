"""Represents the best (based on grid search) settings for
LogisticRegression."""

from sklearn.linear_model import LogisticRegression


class LogisticRegressionClf(object):
    def __init__(self):
        self.clf = LogisticRegression(
            C=1.0,
            class_weight={0: 0.45, 1: 0.55},
            fit_intercept=True,
            max_iter=100,
            penalty='l1',
            solver='newton-cg')

    def get_classifier(self):
        return self.clf
