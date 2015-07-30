from __future__ import print_function

import pandas as pd
from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

vect_feats = None

vectorizer = CountVectorizer(analyzer='word',
                             min_df=3,
                             ngram_range=(1, 1),
                             lowercase=True,
                             strip_accents='ascii')

class NGramsTransformer(TransformerMixin):

    def __init__(self, use=True):
        self.use = use

    def transform(self, features, **transform_params):
        if self.use:
            global vect_feats
            if vect_feats is None:
                vect_feats = vectorizer.fit_transform(features.Name)
                features.drop('Name', axis=1, inplace=True)
                return self._get_ngrams_df(vect_feats.A, features)
            test_feats = vectorizer.transform(features.Name)
            features = self._get_ngrams_df(test_feats.A, features)
        features.drop('Name', axis=1, inplace=True)
        return features

    def _get_ngrams_df(self, ngram_features, features):
        ngrams_df = pd.DataFrame(ngram_features,
                                 columns=vectorizer.get_feature_names(),
                                 index=features.index)
        return pd.concat([features, ngrams_df], axis=1)


    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { 'use': self.use }
