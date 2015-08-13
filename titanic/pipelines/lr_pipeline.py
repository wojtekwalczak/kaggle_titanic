from sklearn.pipeline import Pipeline

from titanic.pipelines.PipelineGetter import PipelineGetter
from titanic.classifiers.ClassifierGetter import ClassifierGetter

#pipeline_getter = PipelineGetter()
pipeline_getter = PipelineGetter(scaler=True,
                                 surname=True,
                                 data_inspector=False,
                                 pca=True,
                                 pca_n_components=30)

features_pipeline = pipeline_getter.get_pipeline()


pipeline = Pipeline([('features', features_pipeline),
                     ('classifier',  ClassifierGetter('lr').get_classifier())])
