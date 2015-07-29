
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.decomposition import TruncatedSVD

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
    ('pclass_transformer', PClassTransformer()),
    ('embarked_transformer', EmbarkedTransformer()),
    ('family_counter', FamilyCounter()),
    ('cabin_transformer', CabinDummyTransformer(complex=False)),
    ('name_transformer', NameTransformer()),
    ('fare_filler', FareFiller()),
    ('ngram', NGramsTransformer()),
    #('data_inspector', DataInspector()),
    ('scaler', Scaler()),
    #('select_features', SelectKBest(f_classif, k=50)),
    ('svd', TruncatedSVD(n_components=15)),
    ('classifier',  GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                        subsample=.25, max_features=.5))])

