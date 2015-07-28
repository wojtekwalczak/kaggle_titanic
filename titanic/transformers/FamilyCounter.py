from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class FamilyCounter(TransformerMixin):

    def transform(self, features_raw, **transform_params):
        features = features_raw.copy(deep=True)
        Family = features_raw[['SibSp', 'Parch']].apply(lambda x: x[0] + x[1], axis=1)
        features.drop('SibSp', axis=1, inplace=True)
        features.drop('Parch', axis=1, inplace=True)
        return pd.concat([features, pd.DataFrame({'Family': Family})], axis=1)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}