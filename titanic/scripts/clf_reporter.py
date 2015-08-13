
from __future__ import print_function

import sys

from sklearn.metrics import classification_report
from titanic.db.train_test_loader import x_test, x_train, y_test, y_train
from titanic.utils.pick_pipeline import pick_pipeline

pipeline = pick_pipeline(sys.argv)
clf = pipeline.fit(x_train, y_train)
print(classification_report(y_test, clf.predict(x_test)))