
from __future__ import print_function

from sklearn.grid_search import GridSearchCV
from titanic.pipelines.lr_pipeline import pipeline
from titanic.db.loader import features, target

logistic_regression_parameters = {
    'features__pclass_transformer__use': (True, False),
    'features__embarked_transformer__use': (True, False),
    'features__family_counter__use': (True, False),
    'features__cabin_transformer__use': (True, False),
    'features__name_transformer__use': (True, False),
    'features__scaler__use': (True, False),
    'features__svd__n_components': (5, 10, 15, 20),
    'classifier__penalty': ('l1', 'l2'),
    'classifier__C': (0.1, 0.25, 0.5, 0.75, 1.0),
    'classifier__fit_intercept': (True, False),
    'classifier__class_weight': ({ 0: 0.4, 1: 0.6 }, { 0: 0.3, 1: 0.7}, 'auto'),
    'classifier__max_iter': (10, 50),
    'classifier__solver': ('newton-cg', 'lbfgs', 'liblinear'),
}

def main():
    grid_search = GridSearchCV(pipeline,
                               param_grid=logistic_regression_parameters,
                               n_jobs=8,
                               verbose=3)
    grid_search.fit(features, target)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(logistic_regression_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    main()