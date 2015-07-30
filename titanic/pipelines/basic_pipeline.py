
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.decomposition import TruncatedSVD

from titanic.classifiers.ClassifierGetter import ClassifierGetter
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
from titanic.transformers.NGramsTransformer import NGramsTransformer
from titanic.transformers.ColumnPicker import ColumnPicker

pipeline = Pipeline([
    ('column_picker', ColumnPicker()),
    ('sex_transformer', SexTransformer()),
    ('age_filler', AgeFiller()),
    ('pclass_transformer', PClassTransformer(use=False)),
    ('embarked_transformer', EmbarkedTransformer(use=False)),
    ('family_counter', FamilyCounter(use=False)),
    ('cabin_transformer', CabinDummyTransformer(use=False)),
    ('name_transformer', NameTransformer(use=False)),
    ('fare_filler', FareFiller()),
    ('name_ngrams', NGramsTransformer(use=True)),
    ('data_inspector', DataInspector(use=False)),
    ('scaler', Scaler(use=False)),
    #('select_features', SelectKBest(f_classif, k=5)),
    ('svd', TruncatedSVD(n_components=15)),
    ('classifier',  ClassifierGetter('gbc').get_classifier())
])
