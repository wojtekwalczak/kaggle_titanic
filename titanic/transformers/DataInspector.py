from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class DataInspector(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features_raw, **transform_params):
        if self.use:
            with pd.option_context('display.max_rows', 999,
                                   'display.max_columns', 999):
                print(features_raw.columns)
        return features_raw

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }