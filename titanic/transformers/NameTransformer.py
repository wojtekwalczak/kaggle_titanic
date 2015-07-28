from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class NameTransformer(TransformerMixin):

    def transform(self, X_raw, **transform_params):
        features = X_raw.copy(deep=True)
        features.Name = features.Name.apply(lambda x: len(x))
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}