from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class CabinDummyTransformer(TransformerMixin):
    def __init__(self, complex=False):
        self.complex = complex

    def transform(self, features_raw, **transform_params):
        features = features_raw.copy(deep=True)
        features.Cabin = features.Cabin.fillna('Z')
        features.Cabin = features.Cabin.apply(lambda x: 'Z' if x[0] == 'Z'\
                                                            else 'A')
        cabin_dummies = pd.get_dummies(features.Cabin, prefix='Cabin')
        features.drop('Cabin', axis=1, inplace=True)
        return pd.concat([features, pd.DataFrame(cabin_dummies)], axis=1)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {'complex': self.complex}
