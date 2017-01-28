""" Adaptive SVC algorithm """

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.classification import accuracy_score


class AdaptiveSVC:
    """
    This class implements the adaptive SVM algorithm based on the article
    "Detecting Concept Drift with Support Vector Machines" by R Klinkenberg and Thorsten Joachims
    """
    def __init__(self, memory_limit=10000, **svc_kwargs):
        """
        Adaptive SVC constructor
        :parameter memory_limit: limit of line of data kept in memory, integer
        :parameter svc_kwargs: kwargs to give to the SVC classifiers, kwargs
        /!\ Only linear kernel is currently supported, so kwargs kernel-related will not be handled as expected
        """
        self.memory = {'X': list(), 'y': list()}
        self.memory_limit = memory_limit
        self.memory_current_size = 0
        self.windows_in_memory = 0
        self.previous_best_window = 0
        self.svc_kwargs = svc_kwargs
        self.classifiers = list()
        self.training_set_sizes = list()
        self.xi_alpha_estimators = list()
        self.predicting_classifier = None

    def _add_new_batch_to_memory(self, X, y, batch_size):
        """
        Add a new batch of data to the memory
        :parameter X: batch of unlabelled data, numpy.array, shape (n,m)
        :parameter y: labels of the batch of data, numpy.array, shape (n,1)
        :parameter batch_size: size of the batch, or len(y), integer == n
        """

        if self.memory_current_size + batch_size > self.memory_limit:
            # It is not possible to store the last batch without exceeding the memory limit
            # So let is forget some of the oldest data
            number_of_data_to_forget = self.memory_current_size + batch_size - self.memory_limit
            oldest_batch_size = len(self.memory['X'][0])
            if number_of_data_to_forget > oldest_batch_size:
                # More than the oldest batch has to be forgotten
                # Let is pop this oldest batch and remove the necessary amount of data from the second oldest batch
                self.memory['X'].pop(0)
                self.memory['y'].pop(0)
                self.memory_current_size -= oldest_batch_size
                number_of_data_to_forget -= oldest_batch_size
                self.windows_in_memory -= 1
            self.memory['X'][0] = self.memory['X'][0][number_of_data_to_forget:]
            self.memory['y'][0] = self.memory['y'][0][number_of_data_to_forget:]
            self.memory_current_size -= number_of_data_to_forget

            if self.memory['X'][0].size == 0:
                # The entire oldest batch has been forgotten and is now empty
                # Let is remove it from the memory
                self.memory['X'].pop(0)
                self.memory['y'].pop(0)
                self.windows_in_memory -= 1

        assert(self.memory_current_size + batch_size <= self.memory_limit)

        # Add the entire new batch of data to memory
        self.memory['X'].append(X)
        self.memory['y'].append(y)
        self.memory_current_size += batch_size
        self.windows_in_memory += 1

    def _svc_fit_on_window(self, window):
        """
        Return an SVC classifier, fitted on the window given in argument
        :parameter window: number of batches to use from memory for learning, 0 < integer <= self.windows_in_memory
        """
        if window is not None:
            X_train = np.concatenate(self.memory['X'][-window:], axis=0)
            y_train = np.concatenate(self.memory['y'][-window:], axis=0)
            return SVC(**self.svc_kwargs, kernel='linear').fit(X_train, y_train), len(y_train)
        else:
            return None

    def _compute_xi_alpha_estimators(self, X, y, batch_size):
        xi_alpha_estimators = list()
        # Compute R
        gram_X = X.dot(X.T)
        diag_gram_X = np.diag(gram_X).reshape(len(gram_X), 1)
        R = np.max(diag_gram_X - gram_X)
        for classifier, training_set_size in zip(self.classifiers, self.training_set_sizes):
            # Compute Xi
            w_opt = classifier.coef_
            b_opt = classifier.intercept_
            xi = np.zeros(shape=(batch_size, 1))
            for data_index, (X_data_index, y_data_index) in enumerate(zip(X, y)):
                xi[data_index] = max(1 - float(y_data_index * (w_opt.dot(X_data_index) + b_opt)), 0)
            # Compute alpha
            alpha = np.zeros(shape=(batch_size, 1))
            for support_vector_idx, alpha_coef in zip(classifier.support_, classifier.dual_coef_):
                alpha_idx = support_vector_idx - training_set_size - batch_size
                if alpha_idx > 0:
                    alpha[alpha_idx] = alpha_coef
            # Compute xi-alpha estimator
            xi_alpha_estimators.append(np.sum(abs(alpha * R + xi) > 1).astype(int) / batch_size)
        return xi_alpha_estimators

    def _update_memory_according_to_best_window(self, window, batch_size):
        """
        Remove unnecessary data from memory according to the chosen best window for learning
        :parameter window: chosen best window for learning, 0 < integer <= self.windows_in_memory
        :parameter batch_size: size of the current batch, integer
        """
        if window < self.windows_in_memory:
            oldest_batch_size = len(self.memory['X'][0])
            self.memory['X'] = self.memory['X'][-window:]
            self.memory['y'] = self.memory['y'][-window:]
            self.memory_current_size -= oldest_batch_size
            if window < self.windows_in_memory - 1:
                self.memory_current_size -= (self.windows_in_memory - 1 - window) * batch_size
            self.windows_in_memory = window

    def update(self, X, y):
        """
        Update the model with the batch given in argument
        :parameter X: batch of unlabelled data, numpy.array, shape (n,m)
        :parameter y: labels of the batch of data, numpy.array, shape (n,1)
        """
        # Add this new batch to the memory
        batch_size = len(y)
        self._add_new_batch_to_memory(X, y, batch_size)

        # Learn on the different windows
        self.classifiers = list()
        self.training_set_sizes = list()
        training_windows = range(1, (self.previous_best_window + 1) + 1)

        for training_window in training_windows:
            window_is_admissible = 0 < training_window <= self.windows_in_memory
            if window_is_admissible:
                clf, training_set_size = self._svc_fit_on_window(training_window)
                self.classifiers.append(clf)
                self.training_set_sizes.append(training_set_size)

        # Compute the xi alpha estimators
        self.xi_alpha_estimators = self._compute_xi_alpha_estimators(X, y, batch_size)

        # Keep the classifier which has the best xi-alpha estimator
        best_classifier_index = np.argmin(self.xi_alpha_estimators)

        # Update self.previous_best_window
        assert(0 < training_windows[best_classifier_index] <= self.windows_in_memory)
        best_window = training_windows[best_classifier_index]
        self.previous_best_window = best_window

        # Update memory
        self._update_memory_according_to_best_window(best_window, batch_size)

        # Update predicting classifier
        self.predicting_classifier = self.classifiers[best_classifier_index]

    def predict(self, X):
        """
        Predict labels associated to data given in argument
        :parameter X: batch of unlabelled data, numpy.array, shape (n,m)
        """
        # Predict with the predicting classifier
        if self.predicting_classifier is not None:
            return self.predicting_classifier.predict(X)
        else:
            return np.zeros(shape=(len(X), 1))


if __name__ == "__main__":
    from StreamGenerator import StreamGenerator
    from DataLoader import SEALoader

    # generate data
    sea_loader = SEALoader('data/sea.data')
    sea_generator = StreamGenerator(sea_loader)

    # model
    clf = AdaptiveSVC(memory_limit=5000, C=100)

    for i, (X, y) in enumerate(sea_generator.generate(batch=2000)):
        print("\nBatch #%d:" % i)
        print("Update model")
        clf.update(X, y)
        print("clf.previous_best_window:", clf.previous_best_window)
        print("clf.training_set_sizes:", clf.training_set_sizes)
        print("clf.xi_alpha_estimators:", clf.xi_alpha_estimators)
        print("clf.windows_in_memory:", clf.windows_in_memory)
        print("clf.memory_current_size:", clf.memory_current_size)
        print("clf.memory_limit:", clf.memory_limit)
        # predict
        print("Predict for current X")
        y_predict = clf.predict(X)
        print("Accuracy score: %0.2f" % accuracy_score(y, y_predict))
        if i > 9:
            break
