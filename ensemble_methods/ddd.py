from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score

from drift_detection_methods.spc import DDM
from ensemble_methods.bagging import OnlineBagging


class PrequentialMetrics:
    def __init__(self):
        self.acc = 1
        self.var = 0
        self.std = 0
        self.t = 0  # time_step
        self.t_drift = 0  # time step of the previous drift

    def update(self, y_pred, y_true, drift):
        """
        Update the Prequential accuracy according to the section 5 of the DDD publication
        if drift
        acc(t) = acc_ex(t)
        else
        acc(t) = acc(t-1) + acc_ex(t)-acc(t-1)/(t -t_drift+1)
        :param y_pred: predicted labels
        :param y_true: real labels
        :param drift: A drift has been detected
        :return:
        """

        number_of_time_steps = len(y_pred)  # number of time steps in the batch
        self.t += number_of_time_steps  # update the number of items seen
        good_predictions = np.sum(y_pred == y_true)
        batch_accuracy = good_predictions / number_of_time_steps

        if drift:
            self.acc = batch_accuracy
            self.var = self.acc * (1 - self.acc) / number_of_time_steps
            self.t_drift = self.t
        else:
            self.acc += (batch_accuracy - self.acc) / (self.t - self.t_drift + 1)
            self.var = self.acc * (1 - self.acc) / (self.t - self.t_drift + 1)

        self.std = np.sqrt(self.var)


class DDD:
    def __init__(self, drift_detector=None, ensemble_method=None, W=0.1, pl=None, ph=None, pd=None):
        '''
        This class implements the DDD algorithms based on the article:
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

        if drift_detector is None:
            self.drift_detector = DDM
        else:
            self.drift_detector = drift_detector
        if ensemble_method is None:
            self.ensemble_method = OnlineBagging
        else:
            self.ensemble_method = ensemble_method
        self.drift_detector = drift_detector()
        self.W = W
        self.pl = pl
        self.ph = ph

        # Parameters
        self.mode_before_drift = True  # before drift
        self.drift = False
        self.low_diversity_learner, self.high_diversity_learner = self.__init_ensemble()
        self.old_low_diversity_learner = self.old_high_diversity_learner = None
        self.metric_ol, self.metric_oh, self.metric_nl, self.metric_nh = self.__init_metrics()
        self.woh = self.wol = self.wnl = 0
        self.y_pred = None

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
        metric_ol = PrequentialMetrics()
        metric_oh = PrequentialMetrics()
        metric_nl = PrequentialMetrics()
        metric_nh = PrequentialMetrics()
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

    def predict(self, X):
        # Before a drift is detected only the low ensemble is used for system prediction
        if self.mode_before_drift:
            y_pred = self.low_diversity_learner.predict(X)
        else:
            sum_acc = self.metric_nl.acc + self.metric_ol.acc * self.W + self.metric_oh.acc
            self.wnl = self.metric_nl.acc / sum_acc
            self.wol = self.metric_ol.acc * self.W / sum_acc
            self.woh = self.metric_oh.acc / sum_acc
            y_pred = self.__weighted_majority(X, self.low_diversity_learner, self.old_low_diversity_learner,
                                              self.old_high_diversity_learner, self.wnl, self.wol, self.woh)
        self.y_pred = y_pred
        return y_pred

    def __drift_detection(self, X, y_true):
        # Not done in the paper but seems to be the proper position for the update
        self.metric_nl.update(self.y_pred, y_true, self.drift)
        self.metric_nh.update(self.high_diversity_learner.predict(X), y_true, self.drift)
        if not self.mode_before_drift:
            self.metric_oh.update(self.old_high_diversity_learner.predict(X), y_true, self.drift)
            self.metric_ol.update(self.old_low_diversity_learner.predict(X), y_true, self.drift)

        # Boolean == True if drift detect
        self.drift = self.drift_detector.drift_detection(y_true, self.y_pred)

        if self.drift:
            # The old low diversity ensemble after the second drift detection can be either
            # the same as the old high diversity learning with low diversity
            # after the first detection or the ensemble corresponding
            # to the new low diversity after the first drift detection depending
            # on which of them is the most accurate.
            if self.mode_before_drift or (not self.mode_before_drift and self.metric_nl.acc > self.metric_oh.acc):
                self.old_low_diversity_learner = self.low_diversity_learner
                self.metric_ol = self.metric_nl  # Not said in the paper but make sense.
            else:
                self.old_low_diversity_learner = self.old_high_diversity_learner
                self.metric_ol = self.metric_oh  # Not said in the paper but make sense.

            # The ensemble corresponding to the high diversity is registered as old
            self.old_high_diversity_learner = self.high_diversity_learner
            self.metric_oh = self.metric_nh  # Not said in the paper but make sense.

            # After a drift is detected new low and high diversity ensemble are created
            self.low_diversity_learner, self.high_diversity_learner = self.__init_ensemble()
            # In the paper all the metrics are set to zero. Which is impossible in the predict method we divide
            # by 0.
            _, _, self.metric_nl, self.metric_nh = self.__init_metrics()
            self.mode_before_drift = False  # After drift
        # if after drift
        if not self.mode_before_drift:
            if self.metric_nl.acc > self.metric_oh.acc and self.metric_nl.acc > self.metric_ol.acc:
                self.mode_before_drift = True
            elif self.metric_oh.acc - self.metric_oh.std > self.metric_nl.acc + self.metric_nl.std \
                    and self.metric_oh.acc - self.metric_oh.std > self.metric_ol.acc + self.metric_ol.std:
                self.low_diversity_learner = deepcopy(self.old_high_diversity_learner)
                self.metric_nl = deepcopy(self.metric_oh)
                self.mode_before_drift = True

    def update(self, X, y_true):
        # If we have never done predictions we cannot detect if there was a drift.
        if self.y_pred is not None:
            self.__drift_detection(X, y_true)
        self.low_diversity_learner.update(X, y_true)
        self.high_diversity_learner.update(X, y_true)
        if not self.mode_before_drift:
            self.old_low_diversity_learner.update(X, y_true)
            self.old_high_diversity_learner.update(X, y_true)

if __name__ == "__main__":
    from data_management.StreamGenerator import StreamGenerator
    from data_management.DataLoader import KDDCupLoader, SEALoader
    from sklearn.linear_model import LogisticRegression

    # generate data
    loader = SEALoader('../data/sea.data', percentage_historical_data=0.1)
    generator = StreamGenerator(loader)
    # kdd_data_loader = KDDCupLoader('../data/kddcup.data_10_percent')
    # generator = StreamGenerator(kdd_data_loader)

    # model
    clf = OnlineBagging
    PARAM_LOG_REG = {'solver': 'sag', 'tol': 1e-1, 'C': 1e4}
    n_classes = np.array(range(0, 2))
    p_clf_high = {'lambda_diversity': 0.1,
                  'n_classes': n_classes,
                  'n_estimators': 25,
                  'base_estimator': LogisticRegression,
                  'p_estimators': PARAM_LOG_REG}
    p_clf_low = {'lambda_diversity': 1,
                 'n_classes': n_classes,
                 'n_estimators': 25,
                 'base_estimator': LogisticRegression,
                 'p_estimators': PARAM_LOG_REG}
    ddd = DDD(ensemble_method=clf, drift_detector=DDM, pl=p_clf_low, ph=p_clf_high, pd={'verbose': True})
    batch = 3000
    X_historical, y_historical = generator.get_historical_data()
    ddd.update(X_historical, y_historical)
    for i, (X, y_true) in enumerate(generator.generate(batch_size=batch)):
        y_pred = ddd.predict(X)
        print("Accuracy score: %0.2f" % accuracy_score(y_true, y_pred))
        # after some time, labels are available
        print("update model\n")
        ddd.update(X, y_true)

