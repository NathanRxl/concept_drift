import numpy as np
from sklearn.linear_model import SGDClassifier

PARAM_LOG_REG = {'solver': 'sag', 'tol': 1e-1, 'C': 1.e4}

class OnlineBagging:
    def __init__(self, lambda_diversity=0.1, n_estimators=25, base_estimator=None, p_estimators=None,
                 n_classes=None):
        '''
        Online Bagging similar to offline Bagging but introduce the low and high diversity during the bagging
        :param lambda_diversity: low lambda diversity allows high diversity ensemble whereas high lambda_diversity
        induces low diversity.
        :param n_estimators:  number of estimators for the bagging
        :param base_estimator: Online learning algorithm it should implements partial fit.
         The default value is SGDClassifer.
        :param p_estimators: Parameters of the online learning algorithms
        :param n_classes: number of classes you need to pass a list of classes
        '''
        if base_estimator is None:
            self.base_estimator = SGDClassifier
        else:
            self.base_estimator = base_estimator

        self.lambda_diversity = lambda_diversity
        if p_estimators is not None:
            self.list_classifiers = [self.base_estimator(**p_estimators) for _ in range(n_estimators)]
        else:
            self.list_classifiers = [self.base_estimator() for _ in range(n_estimators)]

        self.list_classes = n_classes

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
            X_training = None
            y_training = None
            while np.sum(k > 0):
                pos = np.where(k > 0)
                if X_training is None and y_training is None:
                    X_training = X[pos]
                    y_training = y[pos]
                else:
                    X_pos = X[pos]
                    y_pos = y[pos]
                    if X_pos.shape[0] == 1:
                        X_training = np.concatenate((X_training, X[pos].reshape((1, X[pos].shape[1]))), axis=0)
                    else:
                        X_training = np.concatenate((X_training, X[pos]), axis=0)
                    y_training = np.vstack((y_training.reshape((-1, 1)), y_pos.reshape((-1, 1))))

                # check if there is all classes pass to the fit methods
                k -= 1
            if X_training is not None and y_training is not None:
                y_training = y_training.reshape((y_training.shape[0],))
                classifier.partial_fit(X_training, y_training, self.list_classes)

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
    from data_management.StreamGenerator import StreamGenerator
    from data_management.DataLoader import SEALoader
    from sklearn.metrics import accuracy_score
    np.random.seed(3)
    # generate data
    loader = SEALoader('../data/sea.data')
    generator = StreamGenerator(loader)

    # model
    n_classes = np.array(range(0, 2))
    clf = OnlineBagging(base_estimator=SGDClassifier, lambda_diversity=0.1, n_classes=n_classes, n_estimators=25,
                        p_estimators=None)
    X_histo, y_histo = generator.get_historical_data()
    clf.update(X_histo, y_histo)
    for i, (X, y) in enumerate(generator.generate(batch_size=50)):
        print("Batch #%d:" % i)
        # predict
        print("predict for current X")
        y_predict = clf.predict(X)
        # probas = clf.predict_proba(X)
        print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))
        # after some time, labels are available
        print("update model\n")
        clf.update(X, y)
