from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class CabinDummyTransformer(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features, **transform_params):
        if self.use:
            features = features.copy(deep=True)
            features.Cabin = features.Cabin.fillna('Z')
            features.Cabin = features.Cabin.apply(lambda x: 'Z' if x[0] == 'Z'\
                                                                else 'A')
            cabin_dummies = pd.get_dummies(features.Cabin, prefix='Cabin')
            features = pd.concat([features, pd.DataFrame(cabin_dummies)], axis=1)
        features.drop('Cabin', axis=1, inplace=True)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }
