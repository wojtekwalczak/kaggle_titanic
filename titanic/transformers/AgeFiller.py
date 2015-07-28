from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

age_median_by_sex_class = None

class AgeFiller(TransformerMixin):

    def transform(self, features, **transform_params):
        """Fills missing age values with medians calculated for
        groups of passengers, based on their sex and Pclass."""
        features_full_age = features.copy(deep=True)
        # we want to count the age median by sex and Pclass
        # only for training data, and use it for test data
        global age_median_by_sex_class
        if age_median_by_sex_class is None:
            age_median_by_sex_class = features.groupby(['Sex_female',
                                                        'Pclass']).Age.median()

        for i, line in features.iterrows():
            if pd.isnull(line.Age):
                age = age_median_by_sex_class['Sex_female'==line.Sex_female]\
                    .get_value(int(line.Pclass))
                features_full_age.loc[i, 'Age'] = age
        return features_full_age

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
