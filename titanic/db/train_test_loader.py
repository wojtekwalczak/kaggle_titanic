from __future__ import print_function
from sklearn.cross_validation import train_test_split
from titanic.db.loader import features, target

x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.05)

print('x_train: {}; x_test: {}; y_train: {}; y_test: {}'.format(
    x_train.shape, x_test.shape, y_train.shape, y_test.shape))
