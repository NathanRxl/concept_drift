from sklearn.base import ClassifierMixin


class OfflineAlgorithmsWrapper:
    """ Wrapper on Scikit-learn classifiers, to use offline algorithms inside the project """

    def __init__(self, base_estimator):
        """ Constructor of OfflineAlgorithmsWrapper

        :param base_estimator: instance of a classifier of scikit-learn (must be an instance of a subclass of sklearn.base.ClassifierMixin)
        """
        self.base_estimator = base_estimator
        self._check_base_estimator()

        self.fitted = False  # boolean which is True if base_estimator has been fit

    def _check_base_estimator(self):
        """ Raise a ValueError if base_estimator is not suitable for the project """
        if not isinstance(self.base_estimator, ClassifierMixin):
            raise ValueError(
                "In constructor of OfflineAlgorithmsWrapper, base_estimator should be an instance of a subclass of sklearn.base.ClassifierMixin")

    def update(self, X, y):
        """ Fit the base_estimator, only if it has not been fitted already"""
        if not self.fitted:
            self.base_estimator.fit(X, y)
            self.fitted = True

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


# Code example to test OfflineAlgorithmsWrapper
if __name__ == "__main__":
    import numpy as np
    from data_management import SEALoader, StreamGenerator
    from sklearn.ensemble import RandomForestClassifier

    # generate data
    print("Get data...")
    loader = SEALoader('../data/sea.data')
    generator = StreamGenerator(loader)
    X_train, y_train = generator.get_historical_data()
    X_test, y_test = generator.generate(batch_size=1000).__next__()

    # create some models
    clf = OfflineAlgorithmsWrapper(RandomForestClassifier(n_estimators=100))

    # fit models
    print("\nValue of self.fitted: %d" % clf.fitted)
    print("First call of update()...")
    clf.update(X_train, y_train)
    print("Value of self.fitted: %d" % clf.fitted)

    # predict
    print("\nPrediction of classes...")
    y_predict1 = clf.predict(X_test)

    # predict_proba
    print("\nPrediction of probabilities...")
    y_probas1 = clf.predict_proba(X_test)

    # try to update on X_test, y_test
    print("\nSecond call of update()...")
    clf.update(X_test, y_test)

    # second prediction
    print("\nSecond prediction of probabilities...")
    y_probas2 = clf.predict_proba(X_test)

    # comparison of probabilities computed after first and second call to update
    probs_are_equal = np.all(y_probas1 == y_probas2)
    if probs_are_equal:
        print("\nProbabilities are equal after first and second call to update")
        print("  -> OfflineAlgorithmsWrapper works as intended")
    else:
        print("\nProbabilities are different after first and second call to update")
        print("  -> OfflineAlgorithmsWrapper doesn't work as intended")
