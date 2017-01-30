import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class AlgorithmsComparator:
    def __init__(self, algorithms, stream_generator):
        """ Constructor of AlgorithmsComparator

        :param algorithms: iterable of tuples (algorithm_name, algorithm)
        :param stream_generator: instance of StreamGenerator
        """

        self.algorithms = algorithms
        self.stream_generator = stream_generator

        self.predictions = dict()
        self.accuracies = defaultdict(list)
        self.precisions = defaultdict(list)
        self.recalls = defaultdict(list)
        self.f1_scores = defaultdict(list)
        self.X = None
        self.y = None

        self.time_to_update = defaultdict(list)
        self.time_to_predict = defaultdict(list)

    def _set_batch(self, X, y):
        """ Set X and y for current batch"""
        self.X = X
        self.y = y

    def _update_algorithms(self):
        """ Update algorithms with self.X and self.y """
        for algorithm_name, algorithm in self.algorithms:
            print("\t\tAlgorithm {} ... ".format(algorithm_name), end="", flush=True)
            start_timer = time.clock()
            algorithm.update(self.X, self.y)
            end_timer = time.clock()
            time_to_update = end_timer - start_timer
            self.time_to_update[algorithm_name].append(time_to_update)
            print("OK. Time to update on this batch: {0:.3f} seconds".format(time_to_update))

    def _predict_algorithms(self):
        """ Make the predictions for each algorithm on self.X"""
        for algorithm_name, algorithm in self.algorithms:
            print("\t\tAlgorithm {} ... ".format(algorithm_name), end="", flush=True)
            start_timer = time.clock()
            self.predictions[algorithm_name] = algorithm.predict(self.X)
            end_timer = time.clock()
            time_to_predict = end_timer - start_timer
            self.time_to_predict[algorithm_name].append(time_to_predict)
            print("OK. Time to predict on this batch: {0:.3f} seconds".format(time_to_predict))

    def _evaluate_algorithms(self):
        """ Evaluate the performance of the algorithms on current batch"""
        for algorithm_name, algorithm in self.algorithms:
            # compute metrics
            accuracy = accuracy_score(self.y, self.predictions[algorithm_name])
            precision = precision_score(self.y, self.predictions[algorithm_name])
            recall = recall_score(self.y, self.predictions[algorithm_name])
            f1 = f1_score(self.y, self.predictions[algorithm_name])

            # add scores to dictionaries
            self.accuracies[algorithm_name].append(accuracy)
            self.precisions[algorithm_name].append(precision)
            self.recalls[algorithm_name].append(recall)
            self.f1_scores[algorithm_name].append(f1)

    def _plot(self, show_plot):
        """ Create the different plots """
        # create 2*2 subplots
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        accuracy_fig = ax[0, 0]
        precision_fig = ax[0, 1]
        recall_fig = ax[1, 0]
        f1_fig = ax[1, 1]

        for algorithm_name, algorithm in self.algorithms:
            accuracy_fig.plot(self.accuracies[algorithm_name], label=algorithm_name)
            precision_fig.plot(self.precisions[algorithm_name], label=algorithm_name)
            recall_fig.plot(self.recalls[algorithm_name], label=algorithm_name)
            f1_fig.plot(self.f1_scores[algorithm_name], label=algorithm_name)

        # set title
        accuracy_fig.set_title("Accuracies over time")
        precision_fig.set_title("Precisions over time")
        recall_fig.set_title("Recalls over time")
        f1_fig.set_title("F1 scores over time")

        # locate legend
        accuracy_fig.legend(loc="best")
        precision_fig.legend(loc="best")
        recall_fig.legend(loc="best")
        f1_fig.legend(loc="best")

        # set figures' limits
        # accuracy_fig.set_ylim(0.7, 1.0)
        # precision_fig.set_ylim(0.7, 1.0)
        # recall_fig.set_ylim(0.7, 1.0)
        # f1_fig.set_ylim(0.7, 1.0)

        # set x-axis labels
        accuracy_fig.set_xlabel("Batch number")
        precision_fig.set_xlabel("Batch number")
        recall_fig.set_xlabel("Batch number")
        f1_fig.set_xlabel("Batch number")

        # save figure
        plt.savefig("figures/plots_{}.png".format(time.strftime("%Y%m%d_%H%M%S")), format="png")

        # show plot if needed
        if show_plot:
            plt.show()

    def plot_comparison(self, batch_size, stream_length=1e8, show_plot=True):
        """ Main method of AlgorithmsComparator: Simulate data stream and plot the performances of each algorithm"""
        print("\nLet is begin plot_comparison", end="\n" * 2)
        # first training on historical data
        X_train, y_train = self.stream_generator.get_historical_data()
        print("Historical Data")
        self._set_batch(X_train, y_train)
        print("\tUpdate on historical data")
        self._update_algorithms()

        # simulate data streaming
        for batch_nb, (X, y) in enumerate(
                self.stream_generator.generate(batch_size=batch_size, stream_length=stream_length)):
            print("Batch #{}".format(batch_nb))
            # set current batch's X and y
            self._set_batch(X, y)

            # predict current batch, evaluate the performances and update the algorithms
            print("\tPrediction #{}".format(batch_nb))
            self._predict_algorithms()
            print("\tEvaluation #{}".format(batch_nb))
            self._evaluate_algorithms()
            print("\tUpdate #{}".format(batch_nb))
            self._update_algorithms()

        print("Mean time to update")
        for algorithm_name, _ in self.algorithms:
            print(
                "\t{0}: {1:.3f} seconds".format(algorithm_name, np.mean(np.array(self.time_to_update[algorithm_name]))))

        print("Mean time to predict")
        for algorithm_name, _ in self.algorithms:
            print("\t{0}: {1:.3f} seconds".format(algorithm_name,
                                                  np.mean(np.array(self.time_to_predict[algorithm_name]))))

        # make the plots
        self._plot(show_plot)
