from copy import deepcopy

import numpy as np
from sklearn.linear_model import LogisticRegression

from drift_detection_methods.spc import DDM


class Metrics:
    def __init__(self):
        self.acc = 0
        self.std = 0
        self.errors = 0
        self.ctr = 0  # update the number of items seen

    def update(self, y_pred, y_true):
        self.ctr += len(y_pred)  # update the number of items seen
        self.errors += np.sum(y_pred == y_true)
        self.acc = 1 - self.errors / self.ctr
        # I can write my std as follow because values are just 0 or 1: E(X^2) =E(X)
        self.std = np.sqrt(self.acc * (1 - self.acc))


# I did few modifications from the paper
class DDD:
    def __init__(self, stream, drift_detector=DDM, ensemble_method=LogisticRegression, W=0.1, pl=None, ph=None, pd=None,
                 callback=None, verbose=False):
        '''
        This is an implementation of:
        MINKU, Leandro L. et YAO, Xin. DDD: A new ensemble approach for dealing with concept drift. IEEE transactions on
         knowledge and data engineering, 2012, vol. 24, no 4, p. 619-633.
        :param ensemble_method: online ensemble algorithm (LogisticRegression by default)
        :param drift_detector: drift detection method to use
        :param stream: data stream
        :param W: multiplier constant W for the weight of the old low diversity ensemble
        :param pl: parameters for ensemble learning with low diversity
        :param ph: parameters for ensemble learning with high diversity
        :param pd: parameters for drift detection method
        :return:
        '''

        self.ensemble_method = ensemble_method
        self.drift_detector = drift_detector
        self.drift_detection = drift_detector()
        self.stream = stream
        self.W = W
        self.pl = pl
        self.ph = ph
        self.pd = pd

        # Parameters
        self.mode_before_drift = True  # before drift
        self.hnl, self.hnh = self.__init_ensemble()
        self.hol = self.hoh = None
        self.metric_ol, self.metric_oh, self.metric_nl, self.metric_nh = self.__init_metrics()
        self.woh = self.wol = self.wnl = 0

    def __weighted_majority(self, X, hnl, hol, hoh, wnl, wol, woh):
        '''
        Weighted majority between all the learning algorithms.
        The new high diversity learning algorithm is not considered because it is likely to have low accuracy
        on the new concept.
        :param hnl: new low diversity learning algorithm
        :param hol: old low diversity learning algorithm
        :param hoh: old high diversity learning algorithm
        :param wnl: weights
        :param wol: weights
        :param woh: weights
        :return:
        '''
        y_hnl = hnl.predict_proba(X)
        y_hol = hol.predict_proba(X)
        y_hoh = hoh.predict_proba(X)
        return self.__scores_to_single_label(wnl * y_hnl + wol * y_hol + woh * y_hoh)

    @staticmethod
    def __init_metrics():
        metric_ol = Metrics()
        metric_oh = Metrics()
        metric_nl = Metrics()
        metric_nh = Metrics()
        return metric_ol, metric_oh, metric_nl, metric_nh

    def __init_ensemble(self):
        hnl = self.ensemble_method(**self.pl)  # ensemble low diversity
        hnh = self.ensemble_method(**self.ph)  # ensemble high diversity
        return hnl, hnh

    @staticmethod
    def __scores_to_single_label(scores):
        if len(scores.shape) == 1:
            return (scores > 0).astype(np.int)
        else:
            return scores.argmax(axis=1)

    def predict(self, X, y_true):
        # Before a drift is detected only the low ensemble is used for system prediction
        if self.mode_before_drift:
            y_pred = self.hnl.predict(X)
        else:
            sum_acc = self.metric_nl.acc + self.metric_ol.acc * self.W + self.metric_oh.acc
            self.wnl = self.metric_nl.acc / sum_acc
            self.wol = self.metric_ol.acc * self.W / sum_acc
            self.woh = self.metric_oh.acc / sum_acc
            y_pred = self.__weighted_majority(X, self.hnl, self.hol, self.hoh, self.wnl, self.wol, self.woh)
            self.metric_oh.update(self.hoh.predict(X), y_true)
            self.metric_ol.update(self.hol.predict(X), y_true)
        # Not done in the paper but seems to be the proper position for the update
        self.metric_nl.update(y_pred, y_true)
        self.metric_nh.update(self.hnh.predict(X), y_true)
        return y_pred

    def __drift_detection(self, y_true, y_pred):
        # Boolean == True if drift detect
        drift = self.drift_detection.drift_detection(y_true, y_pred)
        if drift:
            # reset drift detector not used in the paper
            self.drift_detection = self.drift_detector()
            # The old low diversity ensemble after the second drift detection can be either
            # the same as the old high diversity learning with low diversity
            # after the first detection or the ensemble corresponding
            # to the new low diversity after the first drift detection depending
            # on which of them is the most accurate.
            if self.mode_before_drift or (not self.mode_before_drift and self.metric_nl.acc > self.metric_oh.acc):
                self.hol = self.hnl
                self.metric_ol = self.metric_nl  # Not said in the paper but make sense.
            else:
                self.hol = self.hoh
                self.metric_ol = self.metric_oh  # Not said in the paper but make sense.

            # The ensemble corresponding to the high diversity is registered as old
            self.hoh = self.hnh
            self.metric_oh = self.metric_nh  # Not said in the paper but make sense.

            # After a drift is detected new low and high diversity ensemble are created
            self.hnl, self.hnh = self.__init_ensemble()
            # In the paper all the metrics are set to zero. Which is impossible in the predict method we divide
            # by 0.
            _, _, self.metric_nl, self.metric_nh = self.__init_metrics()
            # We just set to 0 the counter of errors and count of oh and ol it will be update later on
            self.metric_ol.ctr = 0
            self.metric_oh.ctr = 0
            self.metric_ol.errors = 0
            self.metric_oh.errors = 0

            self.mode_before_drift = False  # After drift
        # if after drift
        if not self.mode_before_drift:
            if self.metric_nl.acc > self.metric_oh.acc and self.metric_nl.acc > self.metric_ol.acc:
                self.mode_before_drift = True
            elif self.metric_oh.acc - self.metric_oh.std > self.metric_nl.acc + self.metric_nl.std \
                    and self.metric_oh.acc - self.metric_oh.std > self.metric_ol.acc + self.metric_ol.std:
                self.hnl = deepcopy(self.hoh)
                self.metric_nl = deepcopy(self.metric_oh)
                self.mode_before_drift = True

    def fit(self, X, y_true):
        self.hnl.fit(X, y_true)
        self.hnh.fit(X, y_true)
        if not self.mode_before_drift:
            self.hol.fit(X, y_true)
            self.hoh.fit(X, y_true)

    def run(self, n_estimators=1500):
        # Take the next value of the generator
        # TODO enhance
        batch = 200
        for i, (X, y_true) in enumerate(self.stream(batch=batch)):
            if i * batch < n_estimators:
                self.fit(X, y_true)
            else:
                y_pred = self.predict(X, y_true)
                self.__drift_detection(y_true, y_pred)
                self.fit(X, y_true)


if __name__ == "__main__":
    from StreamGenerator import StreamGenerator
    from DataLoader import SEALoader

    # generate data
    loader = SEALoader('../../data/sea.data')
    generator = StreamGenerator(loader)

    # model
    clf = LogisticRegression
    clf_param = {'solver': 'sag', 'tol': 1e-1, 'C': 1.e4}
    ddd = DDD(generator.generate, ensemble_method=clf, drift_detector=DDM, pl=clf_param, ph=clf_param, pd=None)
    ddd.run()
