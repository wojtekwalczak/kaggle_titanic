
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from titanic.transformers.Scaler import Scaler
from titanic.transformers.DataInspector import DataInspector
from titanic.transformers.CabinDummyTransformer import CabinDummyTransformer
from titanic.transformers.AgeFiller import AgeFiller
from titanic.transformers.EmbarkedTransformer import EmbarkedTransformer
from titanic.transformers.SexTransformer import SexTransformer
from titanic.transformers.ColumnDropper import ColumnDropper
from titanic.transformers.FareFiller import FareFiller
from titanic.transformers.NameTransformer import NameTransformer
from titanic.transformers.FamilyCounter import FamilyCounter

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import TruncatedSVD

pipeline = Pipeline([
    ('family_counter', FamilyCounter()),
    ('name_transformer', NameTransformer()),
    ('fare_filler', FareFiller()),
    ('sex_transformer', SexTransformer()),
    ('age_filler', AgeFiller()),
    ('embarked_transformer', EmbarkedTransformer()),
    ('cabin_transformer', CabinDummyTransformer(complex=False)),
    ('column_dropper', ColumnDropper()),
    #('data_inspector', DataInspector()),
    ('scaler', Scaler()),
    #('select_features', SelectKBest(f_classif, k=30)),
    ('svd', TruncatedSVD(n_components=10)),
    #('classifier', LogisticRegression())
    ('classifier', LogisticRegression(penalty='l2'))
    #('classifier', RandomForestClassifier(n_estimators=100)),
    # ('classifier',  GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
    #                                     subsample=.25, max_features=.5))
     #RandomForestClassifier(n_estimators=100))
    # ('classifier', GradientBoostingClassifier(n_estimators=100,
    #                                           learning_rate=0.1,
	 #                                          min_samples_leaf=1,
	 #                                          min_samples_split=3,
	 #                                          subsample=0.5))
])