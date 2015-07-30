
from __future__ import print_function

from sklearn.grid_search import GridSearchCV
from titanic.pipelines.basic_pipeline import pipeline
from titanic.db.loader import features, target

gradient_boosting_parameters = {
    'classifier__loss': ('exponential',),# ('deviance', 'exponential'),
    'classifier__learning_rate': (0.001,),#, 0.1, 0.25, 0.5, 0.9),
    'classifier__n_estimators': (500,),#(100, 250, 500),
    'classifier__max_depth': (2, 3, 4, 5),
    'classifier__subsample': (0.2, 0.5, 1.0),
    'classifier__min_samples_leaf': (1, 2, 3),
    'classifier__min_samples_split': (1, 2, 3),
    'classifier__max_features': ('log2',),# ('auto', 'sqrt', 'log2'),
    'classifier__warm_start': (True, False)
}

def main():
    grid_search = GridSearchCV(pipeline,
                               param_grid=gradient_boosting_parameters,
                               n_jobs=8,
                               verbose=3)
    grid_search.fit(features, target)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(gradient_boosting_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    main()