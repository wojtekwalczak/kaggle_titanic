
from sklearn.base import TransformerMixin

class FareFiller(TransformerMixin):

    def transform(self, X_raw, **transform_params):
        features = X_raw.copy(deep=True)
        features.Fare = X_raw.Fare.fillna(features.Fare.mean())
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
