
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

class Scaler(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features_raw, **transform_params):
        if self.use:
            return scaler.fit_transform(features_raw)
        return features_raw

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }
