from collections import defaultdict
from sklearn.metrics import accuracy_score

from ensemble_methods import SEA


class AlgorithmsComparator:
    def __init__(self, algorithms, stream_generator):
        """ Constructor of AlgorithmsComparator

        :param algorithms: iterable of tuples (algorithm_name, algorithm)
        :param stream_generator: instance of StreamGenerator
        """

        self.algorithms = algorithms
        self.stream_generator = stream_generator

        self.predictions = dict()
        self.performances = defaultdict[list]
        self.X = None
        self.y = None

    def _set_batch(self, X, y):
        """ Set X and y for current batch"""
        self.X = X
        self.y = y

    def _update_algorithms(self):
        """ Update algorithms with self.X and self.y """
        for algorithm_name, algorithm in self.algorithms:
            algorithm.update(self.X, self.y)

    def _predict_algorithms(self):
        """ Make the predictions for each algorithm on self.X"""
        for algorithm_name, algorithm in self.algorithms:
            self.predictions[algorithm_name] = algorithm.predict(self.X)

    def _evaluate_algorithms(self):
        """ Evaluate the performance of the algorithms on current batch"""
        for algorithm_name, algorithm in self.algorithms:
            # TODO: add other metrics
            performance = accuracy_score(self.y, self.predictions)
            self.performances[algorithm_name].append(performance)

    def _plot(self):
        """ Create the different plots """
        # TODO: fill function
        pass

    def plot_comparison(self, batch_size, stream_length=1e8):
        """ Main method of AlgorithmsComparator: Simulate data stream and plot the performances of each algorithm"""
        # first training on historical data
        X_train, y_train = self.stream_generator.get_historical_data()
        self._set_batch(X_train, y_train)
        self._update_algorithms()

        # simulate data streaming
        for batch_nb, (X, y) in enumerate(
                self.stream_generator.generate(batch_size=batch_size, stream_length=stream_length)):
            # set current batch's X and y
            self._set_batch(X, y)

            # predict current batch, evaluate the performances and update the algorithms
            self._predict_algorithms()
            self._evaluate_algorithms()
            self._update_algorithms()

        # make the plots
        self._plot()
