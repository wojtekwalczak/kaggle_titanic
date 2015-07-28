from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class DataInspector(TransformerMixin):

    def transform(self, X, **transform_params):
        with pd.option_context('display.max_rows', 999, 'display.max_columns', 999):
            print(X)

        return X

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}