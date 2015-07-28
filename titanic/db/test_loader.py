# coding=utf-8

from __future__ import print_function

import numpy as np
import pandas as pd

test_data = pd.read_csv('../../data/test.csv', dtype={"Age": np.float64})
