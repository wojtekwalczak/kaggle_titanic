from sklearn.pipeline import Pipeline

from titanic.classifiers.ClassifierGetter import ClassifierGetter
from titanic.pipelines.features_pipeline import features_pipeline

pipeline = Pipeline([('features', features_pipeline),
                     ('classifier',  ClassifierGetter('lr').get_classifier())])
