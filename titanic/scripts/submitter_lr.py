
from titanic.db.test_loader import test_data
from titanic.db.loader import features, target
from titanic.pipelines.lr_pipeline import pipeline

pipeline.fit(features, target)

predictions = pipeline.predict(test_data)

test_data['Survived'] = predictions

result = test_data[['PassengerId', 'Survived']]

result.to_csv('titanic_result.csv', index=False)

