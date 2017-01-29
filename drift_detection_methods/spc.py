import numpy as np

'''
Detectors based on statistical process control.
SPC considers learning as a process and monitors the evolution of this process.
'''


class DDM:
    '''
    This class follows the article:
    Gama, J., Medas, P., Castillo, G., Rodrigues, P.: Learning with drift detection.
    Lecture Notes in Computer Science 3171 (2004)
    '''

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.pmin = 10e8
        self.smin = 10e8
        self.t = 0  # number of examples seen
        self.ctr_bad_predictions = 0
        self.pi = 1  # error-rate
        self.si = 0  # standard deviation
        self.psi = 10e8
        self.ctr = 0

    def reset_after_drift(self):
        self.pmin = 10e8
        self.smin = 10e8
        self.ctr_bad_predictions = 0
        self.t = 0  # number of examples seen
        self.pi = 1  # error-rate
        self.si = 0  # standard deviation
        self.psi = 10e8

    def __update(self, y_true, y_pred):
        number_of_time_steps = len(y_pred)  # number of time steps in the batch
        self.t += number_of_time_steps  # update the number of items seen

        self.ctr += number_of_time_steps
        good_predictions = np.sum(y_pred == y_true)
        error_rate = 1 - good_predictions / number_of_time_steps
        self.pi += (error_rate - self.pi) / self.t
        # self.ctr_bad_predictions += len(y_true) - np.sum(y_true == y_pred)  # number of bad predictions
        #
        # self.pi = self.ctr_bad_predictions / self.t
        self.si = np.sqrt(self.pi * (1 - self.pi) / self.t)

        if self.t > 30 and self.pi + self.si <= self.psi:
            self.pmin = self.pi
            self.smin = self.si
            self.psi = self.si + self.pi

    def drift_detection(self, y_true, y_pred):
        self.__update(y_true, y_pred)
        if self.t > 30 and self.pi + self.si >= self.pmin + 3 * self.smin:
            if self.verbose:
                print('Drift detected: time_step={0}'.format(self.ctr))
            self.reset_after_drift()
            return True
        elif self.pmin + 2 * self.smin <= self.pi + self.si < self.pmin + 3 * self.smin:
            if self.verbose:
                print('Warning a drift may happens: time_step={0}'.format(self.ctr))
            return False
        else:
            return False


# TODO implement EDDM
class EDDM(DDM):
    '''
    This class is an implementation of the following algorithm:
    BAENA-GARCIA, Manuel, DEL CAMPO-ÁVILA, José, FIDALGO, Raúl, et al.
    Early drift detection method
    In : Fourth international workshop on knowledge discovery from data streams. 2006. p. 77-86.
    http://www.cs.upc.edu/~abifet/EDDM.pdf
    '''

    def __init__(self, verbose=False):
        DDM.__init__(self, verbose=verbose)

    def drift_detection(self, y_true, y_pred):
        pass
