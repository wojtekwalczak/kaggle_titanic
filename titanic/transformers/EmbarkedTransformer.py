from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

class EmbarkedTransformer(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features, **transform_params):
        if self.use:
            embarked = pd.get_dummies(features.Embarked, prefix='Embarked')
            features = pd.concat([features,
                                  pd.DataFrame(embarked.Embarked_C),
                                  pd.DataFrame(embarked.Embarked_Q),
                                  pd.DataFrame(embarked.Embarked_S)],
                                 axis=1)
        features.drop('Embarked', axis=1, inplace=True)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }
