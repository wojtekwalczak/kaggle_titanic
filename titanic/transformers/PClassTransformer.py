from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin


class PClassTransformer(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features, **transform_params):
        if self.use:
            features = features.copy(deep=True)
            dummies = pd.get_dummies(features.Pclass, prefix='Pclass')
            features = pd.concat([features, pd.DataFrame(dummies)], axis=1)
            features.drop('Pclass_3', axis=1, inplace=True)
        features.drop('Pclass', axis=1, inplace=True)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }

    def set_params(self, **params):
        if 'use' in params:
            self.use = params.get('use')
