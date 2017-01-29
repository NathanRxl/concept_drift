import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt

class DWM:
    """ This class implements the DWM algorithm based on the article "Dynamic Weighted Majority: A New Ensemble Method for Tracking Concept Drift" by Jeremy Z. Kolter and Marcus A. Maloof """

    def __init__(self, base_estimator=DecisionTreeClassifier(), beta = 0.8, theta = 0.01 , period = 5):
        # For noisy problems, a period parameter can be added
        """ Constructor of DWM

        :param base_estimator: instance of a classifier class (by default sklearn.tree.DecisionTreeClassifier())
        :param beta : multiplier affecting the weight every time a classifier get the prediction wrong
        :param theta: threshold to remove the classifier from the list
        """
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        else:
            self.base_estimator = base_estimator

        self.list_classifiers = []
        self.new_classifier = None
        self.classifier_to_evaluate = None
        self.list_classes = None
        self.weights = []
        self.theta = theta
        self.beta = beta
        self.period = period

    def update(self, X, y, delete):
        print(delete)
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
        if len(self.list_classifiers) == 0:
            self.list_classifiers.append(self.new_classifier)
            self.weights.append(1)
        # Otherwise, we lower the weights on the lower classifiers, multiplying them by beta
        # Once the weights are lowered, we remove the classifiers with weights under the threshold theta
        elif delete:
            # On each update, we'll use two empty lists to store the classifiers/weighs that pass the tests
            # Once the tests are ran on all classifiers/weights, that new list become the main one
            self.newlist_classifiers = []
            self.newWeights = []

            for clf, weight in zip(self.list_classifiers, self.weights):
                # If the prediction is untrue but the classifier still has enough weight, we'll keep him
                print(np.sum(clf.predict(X) != y))
                if np.sum(clf.predict(X) != y) > 250:
                    print(True, weight * (self.beta), weight * (self.beta) > self.theta)
                    if weight * (self.beta) > self.theta:
                        self.newWeights.append(round(weight * (self.beta), 2))
                        self.newlist_classifiers.append(clf)
                else:
                    self.newWeights.append(round(weight, 2))
                    self.newlist_classifiers.append(clf)



            self.weights = deepcopy(self.newWeights)

            self.list_classifiers = deepcopy(self.newlist_classifiers)
            # The step is finished by normalizing the weight vector
            norm = np.max(self.weights)
            self.weights = [weight / norm for weight in self.weights]
            # Now let's vote with the new weights
            # If the decision is still not correct, then we'll add a new classifier


            """
            # make the prediction for each classifier
            predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])

            # for each class, count the number of times the class is predicted
            nb_votes_by_class = []
            for i in range(self.list_classes):
                nb_votes_by_class.append(0)
                for j in range(len(self.list_classes)):
                    if predictions[j] == self.list_classes[i]:
                        nb_votes_by_class[i] += self.weights[j]
            """
            # for each example, return the class which was predicted the most
            # If the prediction is incorrect, then add a new classifier


        if np.any(self.predict(X) != y):

            # Train and add the new classifier
            self.new_classifier = deepcopy(self.base_estimator)
            self.new_classifier.fit(X, y)
            self.list_classifiers.append(self.new_classifier)
            # Add the matching weight
            self.weights.append(1)
        print(self.weights)
        """
        for clf, weight in zip(self.list_classifiers, self.weights):
            # If the prediction is untrue but the classifier still has enough weight, we'll keep him
            print(self.weights, (self.beta) ** np.sum(clf.predict(X) != y),
                  weight * ((self.beta) ** np.sum(clf.predict(X) != y)))
        """

    """
    def predict(self, X):
        # Make the prediction

        # :param X: examples to predict
        # :return: the prediction y_predict
        # make the prediction for each classifier
        predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])

        # for each class, count the number of times the class is predicted
        nb_votes_by_class = []
        for c in self.list_classes:
            nb_votes_by_class.append(0)
            for prediction, weight in zip(predictions, self.weights):
                print(prediction, c)
                if prediction == c:
                    nb_votes_by_class[len(nb_votes_by_class)] += weight

        # for each example, return the class which was predicted the most
        return self.list_classes[np.argmax(nb_votes_by_class, axis=0)]
    """

    def predict(self, X):
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
        probs =  np.average(array_probas, axis=2, weights = self.weights)
        return self.list_classes[np.argmax(probs, axis = 1)]



if __name__ == "__main__":
    from generator import SEALoader, Generator
    from sklearn.svm import SVC

    # generate data
    loader = SEALoader('../../data/sea.data')
    generator = Generator(loader)

    # model
    beta = 0.50
    theta = 0.1
    period = 3
    clf = DWM(base_estimator=SVC(probability = True), beta = beta, theta = theta, period = period)

    # record scores
    accuracy_results = []

    for i, (X, y) in enumerate(generator.generate(batch=2000)):
        print("Batch #%d:" % i)
        print("update model\n")
        delete = i % period != 0
        clf.update(X, y, delete = i % period == 0)
        # predict
        print("predict for current X")
        y_predict = clf.predict(X)
        print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))
        accuracy_results.append(accuracy_score(y, y_predict))

    plt.plot(accuracy_results)
    plt.ylabel('Accuracy Results')
    plt.show()