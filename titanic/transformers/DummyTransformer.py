from __future__ import print_function

from sklearn.base import TransformerMixin

class DummyTransformer(TransformerMixin):

    def fit_transform(self, features, y=None, **fit_params):
        return features

    def transform(self, features, **transform_params):
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}

    def set_params(self, **params):
        return self
