import numpy as np
from sklearn.linear_model import LogisticRegression

PARAM_LOG_REG = {'solver': 'sag', 'tol': 1e-1, 'C': 1.e4}


class OnlineBagging:
    def __init__(self, lambda_diversity=0.1, n_estimators=25, base_estimator=None, p_estimators=PARAM_LOG_REG,
                 n_classes=None):
        '''
        Online Bagging similar to offline Bagging but introduce the low and high diversity during the bagging
        :param lambda_diversity: low lambda diversity allows high diversity ensemble whereas high lambda_diversity
        induces low diversity.
        :param n_estimators:  number of estimators for the bagging
        :param base_estimator: Online learning algorithm
        :param p_estimators: Parameters of the online learning algorithms
        :param n_classes: number of classes you need to pass a list of classes
        '''
        if base_estimator is None:
            self.base_estimator = LogisticRegression
        else:
            self.base_estimator = base_estimator

        self.lambda_diversity = lambda_diversity
        self.list_classifiers = [LogisticRegression(**p_estimators) for _ in range(n_estimators)]
        self.list_classes = n_classes

    def preprocess_X_and_y_fit(self, X, y):
        y_values = np.unique(y)
        if len(y_values) == len(self.list_classes):
            return X, y
        else:
            for val in self.list_classes:
                if val not in y_values:
                    X = np.concatenate((X, np.zeros((1, X.shape[1]))), axis=0)
                    y = np.vstack((y.reshape((-1, 1)), val))
            return X, y.reshape((y.shape[0],))

    def update(self, X, y):
        """ Update the ensemble of models

        :param X: new single X
        :param y: new single y
        """
        # retrieve list of different classes if it is the first time we fit data
        if self.list_classes is None:
            self.list_classes = np.unique(y)
        for classifier in self.list_classifiers:
            # Generate the number of time I want my classifier see the example
            k = np.random.poisson(self.lambda_diversity, len(X))
            while np.sum(k > 0):
                pos = np.where(k > 0)
                # check if there is all classes pass to the fit methods
                X_pos, y_pos = self.preprocess_X_and_y_fit(X[pos], y[pos])
                classifier.fit(X_pos, y_pos)
                k -= 1

    def predict(self, X):
        """ Make the prediction

        :param X: examples to predict
        :return: the prediction y_predict
        """
        # make the prediction for each classifier
        predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])

        # for each class, count the number of times the class is predicted
        nb_votes_by_class = []
        for c in self.list_classes:
            nb_votes_by_class.append(np.sum(predictions == c, axis=0))

        # for each example, return the class which was predicted the most
        return self.list_classes[np.argmax(nb_votes_by_class, axis=0)]

    def predict_proba(self, X):
        """ Compute the probability of belonging to each class

        :param X: examples to predict
        :return: the probabilities, array of shape (n_examples, n_classes)
        """
        # create empty array to retrieve
        array_probas = np.zeros((len(X), len(self.list_classes), len(self.list_classifiers)))

        # iterate over the classifiers and add the probabilities to the previous array
        for i, clf in enumerate(self.list_classifiers):
            array_probas[:, :, i] = clf.predict_proba(X)

        # compute and return the mean of probas computed by each classifier
        return array_probas.mean(axis=2)


if __name__ == "__main__":
    from StreamGenerator import StreamGenerator
    from DataLoader import SEALoader
    from sklearn.metrics import accuracy_score

    # generate data
    loader = SEALoader('../data/sea.data')
    generator = StreamGenerator(loader)

    # model
    n_estimators = 25
    n_classes = np.array(range(0, 2))
    clf = OnlineBagging(lambda_diversity=1, n_classes=n_classes)
    X_histo, y_histo = generator.get_historical_data()
    clf.update(X_histo, y_histo)
    for i, (X, y) in enumerate(generator.generate(batch_size=500)):
        print("Batch #%d:" % i)
        # predict
        print("predict for current X")
        y_predict = clf.predict(X)
        # probas = clf.predict_proba(X)
        print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))
        # after some time, labels are available
        print("update model\n")
        clf.update(X, y)
