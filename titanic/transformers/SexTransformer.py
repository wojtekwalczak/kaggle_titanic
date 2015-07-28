from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class SexTransformer(TransformerMixin):

    def transform(self, features, **transform_params):
        sex = pd.get_dummies(features.Sex, prefix='Sex')
        features = pd.concat([features,
                              pd.DataFrame(sex.Sex_female)],
                             axis=1)
        features.drop('Sex', axis=1, inplace=True)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}