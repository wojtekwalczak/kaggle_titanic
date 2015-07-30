
from titanic.classifiers.GradiendBoosting import GradiendBoosting

classifiers = { 'gbc': GradiendBoosting }

class ClassifierGetter(object):

    def __init__(self, clf):
        if clf not in classifiers:
            raise KeyError("Classifier {} is unknown".format(clf))
        self.classifier = classifiers.get(clf)()

    def get_classifier(self):
        return self.classifier.get_classifier()
