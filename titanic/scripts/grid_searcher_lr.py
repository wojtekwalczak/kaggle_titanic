
from __future__ import print_function

from sklearn.grid_search import GridSearchCV
from titanic.pipelines.lr_pipeline import pipeline
from titanic.db.loader import features, target

"""
Best score: 0.825
Best parameters set:
   classifier__C: 1.0
	classifier__class_weight: {0: 0.4, 1: 0.6}
	classifier__fit_intercept: False
	classifier__max_iter: 50
	classifier__penalty: 'l1'
	classifier__solver: 'newton-cg'
	features__cabin_transformer__use: True
	features__embarked_transformer__use: True
	features__family_counter__use: True
	features__name_transformer__use: False
	features__pclass_transformer__use: False
	features__scaler__use: False
	features__svd__n_components: 15



Best score: 0.829
Best parameters set:
	classifier__C: 1.0
	classifier__class_weight: {0: 0.5, 1: 0.5}
	classifier__fit_intercept: True
	classifier__max_iter: 10
	classifier__penalty: 'l1'
	classifier__solver: 'newton-cg'
	features__cabin_transformer__use: True
	features__embarked_transformer__use: False
	features__family_counter__use: False
	features__name_transformer__use: True
	features__pca__n_components: 10
	features__pca__whiten: True
	features__pclass_transformer__use: True
	features__scaler__use: False

"""


logistic_regression_parameters = {
    'features__pclass_transformer__use': (True, False),
    'features__embarked_transformer__use': (True, False),
    'features__family_counter__use': (True, False),
    'features__cabin_transformer__use': (True, False),
    'features__name_transformer__use': (True, False),
    'features__scaler__use': (True, False),
    'features__pca__n_components': (5, 10, 15),
    'features__pca__whiten': (True, False),
    #'features__svd__n_components': (5, 10, 15, 20),
    'classifier__penalty': ('l1', 'l2'),
    'classifier__C': (0.1, 1.0, 10, 100),
    'classifier__fit_intercept': (True, False),
    'classifier__class_weight': ({ 0: 0.4, 1: 0.6 }, { 0: 0.3, 1: 0.7}, { 0: 0.5, 1: 0.5 }),
    'classifier__max_iter': (10, 50),
    'classifier__solver': ('newton-cg', 'lbfgs', 'liblinear'),
}

def main():
    grid_search = GridSearchCV(pipeline,
                               param_grid=logistic_regression_parameters,
                               n_jobs=3,
                               verbose=3)
    grid_search.fit(features, target)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(logistic_regression_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    main()