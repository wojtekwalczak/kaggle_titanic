"""Serves instances of pre-defined classifiers."""

from titanic.classifiers.LogisticRegressionClf import LogisticRegressionClf
from titanic.classifiers.GradientBoostingClf import GradientBoostingClf

classifiers = { 'gbc': GradientBoostingClf,
                'lr': LogisticRegressionClf }

class ClassifierGetter(object):

    def __init__(self, clf):
        if clf not in classifiers:
            raise KeyError("Classifier {} is unknown".format(clf))
        self.classifier = classifiers.get(clf)()

    def get_classifier(self):
        return self.classifier.get_classifier()
