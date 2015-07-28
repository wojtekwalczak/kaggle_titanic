
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

class Scaler(TransformerMixin):

    def transform(self, X, **transform_params):
        return scaler.fit_transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
