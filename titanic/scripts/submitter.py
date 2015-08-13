from __future__ import print_function

import sys

from titanic.db.test_loader import test_data
from titanic.db.loader import features, target
from titanic.utils.pick_pipeline import pick_pipeline

pipeline = pick_pipeline(sys.argv)
pipeline.fit(features, target)
test_data['Survived'] = pipeline.predict(test_data)

result = test_data[['PassengerId', 'Survived']]
result.to_csv('titanic_result.csv', index=False)
