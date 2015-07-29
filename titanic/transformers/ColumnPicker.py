
from __future__ import print_function

from sklearn.base import TransformerMixin

class ColumnPicker(TransformerMixin):

    def transform(self, features, **transform_params):
        return features[['Pclass', 'Name', 'Age', 'SibSp',
                         'Parch', 'Fare', 'Cabin',
                         'Embarked', 'Sex']]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { }
