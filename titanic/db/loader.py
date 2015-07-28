# coding=utf-8

from __future__ import print_function

import numpy as np
import pandas as pd

data = pd.read_csv('../../data/train.csv', dtype={"Age": np.float64})

features = data[['Pclass', 'Name', u'Age', u'SibSp', u'Parch', u'Fare', u'Cabin',
                 u'Embarked', u'Sex']]

target = data.Survived.values

