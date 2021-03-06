
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

fare_mean = None

class FareFiller(TransformerMixin):

    def transform(self, features_raw, **transform_params):
        features = features_raw.copy(deep=True)
        global fare_mean
        if fare_mean is None:
            fare_mean = int(features.Fare.mean())
        features.Fare.fillna(fare_mean, axis=0, inplace=True)
        return features


    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
