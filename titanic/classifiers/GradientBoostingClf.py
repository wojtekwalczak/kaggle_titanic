"""Represents the best (based on grid search) settings for
GradientBoostingClassifier."""

from __future__ import print_function

from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingClf(object):
    def __init__(self):
        self.clf = GradientBoostingClassifier(learning_rate=0.01,
                                              loss='exponential',
                                              max_depth=5,
                                              max_features='log2',
                                              min_samples_leaf=2,
                                              min_samples_split=1,
                                              n_estimators=1000,
                                              subsample=0.8,
                                              warm_start=True)

    def get_classifier(self):
        return self.clf
