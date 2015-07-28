
from __future__ import print_function

from sklearn.grid_search import GridSearchCV
from titanic.pipelines.basic_pipeline import pipeline
from titanic.db.train_test_loader import x_test, x_train, y_test, y_train

# parameters = {
#                'classifier__penalty': ('l1', 'l2'),
#                'classifier__C': (0.0001, 0.001, 0.01, 0.1, 1, 10),
#                'classifier__fit_intercept': (True, False),
#                'classifier__class_weight': ('auto', None),
#                'classifier__max_iter': (10, 100),
#              }
#
# loss='deviance', learning_rate=0.1, n_estimators=100,
#                  subsample=1.0, min_samples_split=2,
#                  min_samples_leaf=1, min_weight_fraction_leaf=0.,
#                  max_depth=3, init=None, random_state=None,
#                  max_features=None, verbose=0,
#                  max_leaf_nodes=None, warm_start=False
gradient_boosting_parameters = {
    'classifier__learning_rate': (0.01, 0.1, 0.25, 0.5, 1),
    'classifier__n_estimators': (50, 100, 200),
    'classifier__subsample': (0.2, 0.5, 1.0),
    'classifier__min_samples_leaf': (1, 2, 3),
    'classifier__min_samples_split': (3, 4, 5),
}

def main():
    grid_search = GridSearchCV(pipeline,
                               param_grid=gradient_boosting_parameters)
    grid_search.fit(x_train, y_train)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(gradient_boosting_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    main()