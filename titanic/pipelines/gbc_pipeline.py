from sklearn.pipeline import Pipeline

from titanic.pipelines.PipelineGetter import PipelineGetter
from titanic.classifiers.ClassifierGetter import ClassifierGetter

# with surname=False (and thus, with pca=False as well)
# pipeline_getter = PipelineGetter(scaler=False,
#                                  surname=False,
#                                  data_inspector=False,
#                                  pca=False)

#with surname=True
pipeline_getter = PipelineGetter(scaler=False,
                                 name=True,
                                 embarked=True,
                                 surname=True,
                                 data_inspector=False,
                                 pca=True)

features_pipeline = pipeline_getter.get_pipeline()

pipeline = Pipeline([('features', features_pipeline),
                     ('classifier', ClassifierGetter('gbc').get_classifier())])
