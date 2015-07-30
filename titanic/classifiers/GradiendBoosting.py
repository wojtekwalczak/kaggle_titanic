
from sklearn.ensemble import GradientBoostingClassifier

class GradiendBoosting(object):
    def __init__(self):
        self.clf = GradientBoostingClassifier(learning_rate=0.01,
                                               loss='exponential',
                                               max_depth=5,
                                               max_features='log2',
                                               min_samples_leaf=2,
                                               min_samples_split=1,
                                               n_estimators=300,
                                               subsample=0.8,
                                               warm_start=True)

    def get_classifier(self):
        return self.clf