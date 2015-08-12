from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from titanic.transformers.Scaler import Scaler
from titanic.transformers.DataInspector import DataInspector
from titanic.transformers.CabinDummyTransformer import CabinDummyTransformer
from titanic.transformers.AgeFiller import AgeFiller
from titanic.transformers.EmbarkedTransformer import EmbarkedTransformer
from titanic.transformers.SexTransformer import SexTransformer
from titanic.transformers.FareFiller import FareFiller
from titanic.transformers.NameTransformer import NameTransformer
from titanic.transformers.FamilyCounter import FamilyCounter
from titanic.transformers.PClassTransformer import PClassTransformer
from titanic.transformers.ColumnPicker import ColumnPicker
from titanic.transformers.SurnameTransformer import SurnameTransformer


class PipelineGetter(object):

    def __init__(self, pclass=True, embarked=True, family_count=True,
                 cabin=True, name=True, data_inspector=False, scaler=True,
                 pca_n_components=20, pca_whiten=True):
        self.pipeline = Pipeline([
            ('column_picker', ColumnPicker()),
            ('sex_transformer', SexTransformer()),
            ('age_filler', AgeFiller()),
            ('pclass_transformer', PClassTransformer(use=pclass)),
            ('embarked_transformer', EmbarkedTransformer(use=embarked)),
            ('family_counter', FamilyCounter(use=family_count)),
            ('cabin_transformer', CabinDummyTransformer(use=cabin)),
            ('name_transformer', NameTransformer(use=name)),
            ('fare_filler', FareFiller()),
            ('surname_transformer', SurnameTransformer()),
            ('data_inspector', DataInspector(use=data_inspector)),
            ('scaler', Scaler(use=scaler)),
            ('pca', PCA(n_components=pca_n_components, whiten=pca_whiten))
        ])

    def get_pipeline(self):
        return self.pipeline
