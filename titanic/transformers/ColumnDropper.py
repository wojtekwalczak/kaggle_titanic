from __future__ import print_function

from sklearn.base import TransformerMixin

class ColumnDropper(TransformerMixin):

    def transform(self, features, **transform_params):
        # columns to keep
        columns = ['Pclass', u'Age', 'Sex_female', 'Family', 'Name',
                    u'Fare', u'Sex_female',
                    u'Embarked_C', u'Embarked_Q']
        to_check = ['Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',
                    'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T',
                    'Cabin_no_0', 'Cabin_no_2', 'Cabin_no_3',
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
        for col in to_check:
            if col in features.columns:
                columns.append(col)
        print(columns)
        return features[columns]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, *args, **kwargs):
        return { }
