from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class FamilyCounter(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features_raw, **transform_params):
        if self.use:
            features = features_raw.copy(deep=True)
            family = features_raw[['SibSp', 'Parch']]\
                .apply(lambda x: x[0] + x[1], axis=1)
            features.drop('SibSp', axis=1, inplace=True)
            features.drop('Parch', axis=1, inplace=True)
            return pd.concat([features,
                              pd.DataFrame({'Family': family})], axis=1)
        return features_raw


    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }

    def set_params(self, **params):
        if 'use' in params:
            self.use = params.get('use')