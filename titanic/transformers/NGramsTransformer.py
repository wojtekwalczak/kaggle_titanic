from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = None
vect_feats = None

class NGramsTransformer(TransformerMixin):

    def transform(self, features, **transform_params):

        global vectorizer, vect_feats
        if vectorizer is None:
            vectorizer = CountVectorizer(analyzer='word',
                                         min_df=2,
                                         ngram_range=(1, 1),
                                         lowercase=False)
            vect_feats = vectorizer.fit_transform(features.Name)
            features.drop('Name', axis=1, inplace=True)
            ngrams_df = pd.DataFrame(vect_feats.A,
                                     columns=vectorizer.get_feature_names(),
                                     index=features.index)
            return pd.concat([features, ngrams_df], axis=1)

        feats = vectorizer.transform(features.Name)
        features.drop('Name', axis=1, inplace=True)
        ngrams_df = pd.DataFrame(feats.A,
                                 columns=vectorizer.get_feature_names(),
                                 index=features.index)
        return pd.concat([features, ngrams_df], axis=1)



    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {}
