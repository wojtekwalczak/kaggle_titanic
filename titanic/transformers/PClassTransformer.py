from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin


class PClassTransformer(TransformerMixin):

    def transform(self, features_raw, **transform_params):
        features = features_raw.copy(deep=True)
        dummies = pd.get_dummies(features.Pclass, prefix='Pclass')
        features.drop('Pclass', axis=1, inplace=True)
        return pd.concat([features, pd.DataFrame(dummies)], axis=1)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
