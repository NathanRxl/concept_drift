import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SEA:
    """ This class implements the SEA algorithm based on the article "A Streaming Ensemble Algorithm (SEA) for Large-Scale Classification" by W Nick Street and YongSeog Kim """

    def __init__(self, n_estimators, training_size, base_estimator):
        """ Constructor of SEA

        :param base_estimator: instance of a classifier class (by default sklearn.tree.DecisionTreeClassifier())
        :param n_estimators: number of estimators in the ensemble
        :param training_size: number of examples used to train the classifiers
        """
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier
        else:
            self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.training_size = training_size

        self.list_estimators = []
        self.X = None
        self.y = np.zeros(training_size)

    def update(self, x, y):
        """ Update the ensemble of models

        :param x: new example x
        :param y: label of x
        """
        # update X and y

        # if there is enough example, train a new classifier, evaluate the classifiers, remove the worst performing one
        pass

    def predict(self, x):
        """ Make the prediction

        :param x: new example x
        :return: the prediction
        """
        # make the prediction for example x based on a vote of the ensemble of classifiers
        pass
