# coding=utf-8

from __future__ import print_function

import numpy as np
import pandas as pd

data = pd.read_csv('../../data/train.csv', dtype={"Age": np.float64})

features = data[['Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
                 'Embarked', 'Sex']]

target = data.Survived.values

