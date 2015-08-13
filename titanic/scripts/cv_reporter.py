from __future__ import print_function

import sys
import numpy as np
from sklearn.cross_validation import cross_val_score
from titanic.db.loader import features, target
from titanic.utils.pick_pipeline import pick_pipeline

pipeline = pick_pipeline(sys.argv)
scores = cross_val_score(pipeline, features, target, cv=10)
print(np.mean(scores), np.std(scores))
