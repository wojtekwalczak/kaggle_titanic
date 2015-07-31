
from __future__ import print_function

from sklearn.metrics import classification_report

from titanic.pipelines.features_pipeline import pipeline
from titanic.db.train_test_loader import x_test, x_train, y_test, y_train

clf = pipeline.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test)))