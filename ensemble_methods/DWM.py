import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.classification import accuracy_score


class DWM:
    """ This class implements the DWM algorithm based on the article "Dynamic Weighted Majority: A New Ensemble Method for Tracking Concept Drift" by Jeremy Z. Kolter and Marcus A. Maloof """

    def __init__(self, n_estimators, base_estimator=None, scoring_method=None, beta = None, theta = None ): # For noisy problems, a period parameter can be added
        """ Constructor of DWM

        :param n_estimators: number of estimators in the ensemble
        :param base_estimator: instance of a classifier class (by default sklearn.tree.DecisionTreeClassifier())
        :param beta : multiplier affecting the weight every time a classifier get the prediction wrong
        :param theta: threshold to remove the classifier from the list
        """
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        else:
            self.base_estimator = base_estimator

        if scoring_method is None:
            self.scoring_method = accuracy_score

        self.n_estimators = n_estimators

        self.list_classifiers = []
        self.new_classifier = None
        self.classifier_to_evaluate = None
        self.list_classes = None
        self.weights = [1] * n_estimators
        self.sigma = None
        self.newlist_classifiers = []
        self.newWeights = []

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
        # Otherwise, we lower the weights on the lower classifiers, multiplying them by beta
        # Once the weights are lowered, we remove the classifiers with weights under the threshold theta
        else:
            for i in range(len(self.list_classifiers)):
                clf = self.list_classifiers[i]
                if clf.predict(X) != y and self.weights[i] * self.beta > self.theta:
                    self.newWeights.append(self.weights[i] * self.beta)
                    self.newlist_classifiers.append(self.list_classifiers[i])
                elif clf.predict(X) == y: 
                    self.newWeights.append(self.weights[i])
                    self.newlist_classifiers.append(self.list_classifiers[i])
            self.weights = deepcopy(self.newWeights)
            self.newWeights = []

            self.list_classifiers = deepcopy(self.newlist_classifiers)
            self.newlist_classifiers = []

            # The step is finished by normalizing the weight vector
            norm = max(self.weights)
            weights = [weight / norm for weight in weights]
            # Now let's vote with the new weights
            # If the decision is still not correct, then we'll add a new classifier

            # make the prediction for each classifier
            predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])

            # for each class, count the number of times the class is predicted 
            nb_votes_by_class = []
            for i in range(self.list_classes):
                nb_votes_by_class.append(0)
                for j in range(len(self.list_classes)):
                    if predictions[j] == self.list_classes[i]:
                        nb_votes_by_class[i] += self.weights[j]

            # for each example, return the class which was predicted the most
            # If the prediction is incorrect, then add a new classifier
            if self.list_classes[np.argmax(nb_votes_by_class, axis=0)] != y:

                # Train and add the new classifier
                self.new_classifier = deepcopy(self.base_estimator)
                self.new_classifier.fit(X, y)
                self.list_classifiers.append()

                # Add the matching weight
                self.weight.append(1)

    def predict(self, X):
        """ Make the prediction

        :param X: examples to predict
        :return: the prediction y_predict
        """
        # make the prediction for each classifier
        predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])

        # for each class, count the number of times the class is predicted
        nb_votes_by_class = []
        for i in range(self.list_classes):
            nb_votes_by_class.append(0)
            for j in range(len(self.predictions)):
                if predictions[j] == self.list_classes[i]:
                    nb_votes_by_class[i] += self.weights[j]

        # for each example, return the class which was predicted the most
        return self.list_classes[np.argmax(nb_votes_by_class, axis=0)]


if __name__ == "__main__":
    from generator import SEALoader, Generator
    from sklearn.svm import SVC

    # generate data
    loader = SEALoader('../data/sea.data')
    generator = Generator(loader)

    # model
    beta = 0.5
    theta = 0.01
    n_estimators = 5
    #period = 3
    clf = DWM(base_estimator=SVC(), n_estimators=n_estimators, beta = beta, theta = theta, period = period)

    for i, (X, y) in enumerate(generator.generate(batch=2000)):
        print("Batch #%d:" % i)
        # for the first batches, only update the model
        if i < n_estimators:
            print("update model\n")
            clf.update(X, y)
        else:
            # predict
            print("predict for current X")
            y_predict = clf.predict(X)
            print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))

            # after some time, labels are available
            print("update model\n")
            clf.update(X, y)
