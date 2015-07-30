from __future__ import print_function

import numpy as np
from sklearn.cross_validation import cross_val_score
from titanic.pipelines.basic_pipeline import pipeline
from titanic.db.loader import features, target

scores = cross_val_score(pipeline, features, target, cv=10)
print(np.mean(scores), np.std(scores))