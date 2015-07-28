from __future__ import print_function

import re
import pandas as pd
from sklearn.base import TransformerMixin

cabins = dict(A='A', B='B', C='C', D='D', E='E', F='F', G='G', T='T')
cabin_nos = ['Cabin_no_0', 'Cabin_no_2', 'Cabin_no_3',
             'Cabin_no_4', 'Cabin_no_5', 'Cabin_no_6',
             'Cabin_no_7', 'Cabin_no_8', 'Cabin_no_9',
             'Cabin_no_10', 'Cabin_no_11', 'Cabin_no_12',
             'Cabin_no_14', 'Cabin_no_15', 'Cabin_no_16',
             'Cabin_no_17', 'Cabin_no_18', 'Cabin_no_19',
             'Cabin_no_20', 'Cabin_no_21', 'Cabin_no_22',
             'Cabin_no_23', 'Cabin_no_24', 'Cabin_no_25',
             'Cabin_no_26', 'Cabin_no_28', 'Cabin_no_30',
             'Cabin_no_31', 'Cabin_no_32', 'Cabin_no_33',
             'Cabin_no_34', 'Cabin_no_35', 'Cabin_no_36',
             'Cabin_no_37', 'Cabin_no_38', 'Cabin_no_39',
             'Cabin_no_40', 'Cabin_no_41', 'Cabin_no_42',
             'Cabin_no_44', 'Cabin_no_45', 'Cabin_no_46',
             'Cabin_no_47', 'Cabin_no_48', 'Cabin_no_49',
             'Cabin_no_50', 'Cabin_no_51', 'Cabin_no_52',
             'Cabin_no_54', 'Cabin_no_56', 'Cabin_no_57',
             'Cabin_no_58', 'Cabin_no_62', 'Cabin_no_63',
             'Cabin_no_65', 'Cabin_no_67', 'Cabin_no_68',
             'Cabin_no_69', 'Cabin_no_70', 'Cabin_no_71',
             'Cabin_no_73', 'Cabin_no_77', 'Cabin_no_78',
             'Cabin_no_79', 'Cabin_no_80', 'Cabin_no_82',
             'Cabin_no_83', 'Cabin_no_85', 'Cabin_no_86',
             'Cabin_no_87', 'Cabin_no_90', 'Cabin_no_91',
             'Cabin_no_92', 'Cabin_no_93', 'Cabin_no_94',
             'Cabin_no_95', 'Cabin_no_96', 'Cabin_no_99',
             'Cabin_no_101', 'Cabin_no_102', 'Cabin_no_103',
             'Cabin_no_104', 'Cabin_no_106', 'Cabin_no_110',
             'Cabin_no_111', 'Cabin_no_118', 'Cabin_no_121',
             'Cabin_no_123', 'Cabin_no_124', 'Cabin_no_125',
             'Cabin_no_126', 'Cabin_no_128', 'Cabin_no_148']


def get_cabin_no(cabin_str):
    if not pd.isnull(cabin_str):
        cabins = re.findall('\w(\d+)', cabin_str)
        if cabins:
            return int(cabins[0])
    return 0


def get_cabin_dummies(data_frame):
    dummies = pd.get_dummies(data_frame.Cabin, prefix='Cabin')
    for letter in cabins:
        col_name = 'Cabin_{}'.format(letter)
        if col_name not in dummies.columns and col_name in data_frame:
            dummies[col_name] = 0
    return dummies


def get_cabin_no_dummies(cabin_no_dummies):
    for col_name in cabin_nos:
        if col_name not in cabin_no_dummies.columns:
            cabin_no_dummies[col_name] = 0
    return cabin_no_dummies


class CabinDummyTransformer(TransformerMixin):
    def __init__(self, complex=False):
        self.complex = complex

    def transform(self, X_raw, **transform_params):
        X = X_raw.copy(deep=True)
        Cabin_no = X.Cabin.apply(lambda x: get_cabin_no(x))
        X.Cabin = X.Cabin.fillna('Z')
        if self.complex:
            X.Cabin = X.Cabin.apply(lambda x: 'Z' if x[0] == 'Z' \
                else cabins.get(x[0], 'Z'))
        else:
            X.Cabin = X.Cabin.apply(lambda x: 'Z' if x[0] == 'Z' else 'A')
        cabin_dummies = get_cabin_dummies(X)
        cabin_no_dummies = get_cabin_no_dummies(
            pd.get_dummies(Cabin_no, prefix='Cabin_no'))
        X.drop('Cabin', axis=1, inplace=True)

        return pd.concat([X,
                          pd.DataFrame(cabin_dummies),
                          pd.DataFrame(cabin_no_dummies)],
                         axis=1)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return {'complex': self.complex}
