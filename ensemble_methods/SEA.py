from copy import deepcopy

import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class SEA:
    """
    This class implements the SEA algorithm based on the article
    "A Streaming Ensemble Algorithm (SEA) for Large-Scale Classification" by W Nick Street and YongSeog Kim
    """

    def __init__(self, n_estimators, base_estimator=None, scoring_method=None):
        """ Constructor of SEA

        :param n_estimators: number of estimators in the ensemble
        :param base_estimator: instance of a classifier class (by default sklearn.tree.DecisionTreeClassifier())

        """
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        else:
            self.base_estimator = base_estimator

        if scoring_method is None:
            self.scoring_method = accuracy_score
        else:
            self.scoring_method = scoring_method

        self.n_estimators = n_estimators

        self.list_classifiers = []
        self.new_classifier = None
        self.classifier_to_evaluate = None
        self.list_classes = None

    def update(self, X, y):
        """ Update the ensemble of models

        :param X: new batch X
        :param y: array of labels
        """
        # retrieve list of different classes if it is the first time we fit data
        if self.list_classes is None:
            self.list_classes = np.unique(y)

        # train new classifier
        self.new_classifier = deepcopy(self.base_estimator)
        self.new_classifier.fit(X, y)

        # if there is not enough classifiers, add the new classfier in the ensemble
        if len(self.list_classifiers) < self.n_estimators:
            self.list_classifiers.append(self.new_classifier)
        # otherwise, evaluate the (n_estimators + 1) estimators and remove the worst performing one
        else:
            if self.classifier_to_evaluate is None:
                self.classifier_to_evaluate = self.new_classifier
            else:
                # evaluate (n_estimators + 1) classifiers
                self.list_classifiers.append(self.classifier_to_evaluate)
                scores = [self.scoring_method(y, clf.predict(X)) for clf in self.list_classifiers]

                # remove the worst performing one
                self.list_classifiers.pop(int(np.argmin(scores)))

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
    from data_management import SEALoader, StreamGenerator
    from sklearn.svm import SVC

    # generate data
    loader = SEALoader('../../data/sea.data')
    generator = StreamGenerator(loader)

    # model
    n_estimators = 5
    clf = SEA(base_estimator=SVC(probability=True), n_estimators=n_estimators)

    for i, (X, y) in enumerate(generator.generate(batch_size=2000)):
        print("Batch #%d:" % i)
        # for the first batches, only update the model
        if i < n_estimators:
            print("update model\n")
            clf.update(X, y)
        else:
            # predict
            print("predict for current X")
            y_predict = clf.predict(X)
            # probas = clf.predict_proba(X)
            print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))
            # after some time, labels are available
            print("update model\n")
            clf.update(X, y)





