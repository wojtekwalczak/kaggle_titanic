from __future__ import print_function

import numpy as np
from sklearn.cross_validation import cross_val_score
#from titanic.pipelines.basic_pipeline import pipeline
from titanic.pipelines.basic_pipeline import pipeline
#from titanic.db.train_test_loader import x_test, x_train, y_test, y_train
from titanic.db.loader import features, target
#clf = pipeline.fit(x_train, y_train)


scores = cross_val_score(pipeline, features, target, cv=10)
print(np.mean(scores), np.std(scores))