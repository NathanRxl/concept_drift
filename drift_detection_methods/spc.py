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
        self.__pmin_i = 0  # allows to see when pmin changes
        self.smin = 10e8
        self.__smin_i = 0  # allows to see when smin changes
        self.ctr_bad_pred = 0  # counter of bad predictions
        self.ctr = 0  # number of examples seen
        self.pi = 10e8  # error-rate
        self.si = 10e8  # standard deviation

    def __update(self, y_true, y_pred):
        self.ctr_bad_pred += len(y_true) - np.sum(y_true == y_pred)  # number of bad predictions
        self.ctr += len(y_pred)  # length of the pred
        self.pi = self.ctr_bad_pred / self.ctr  # error-rate = probability of bad predictions at i
        self.si = np.sqrt((self.pi * (1 - self.pi) / self.ctr))  # standard deviation at i

        if self.pi < self.pmin:
            self.pmin = self.pi
            self.__pmin_i = self.ctr
        if self.si < self.smin:
            self.smin = self.si
            self.__smin_i = self.ctr  # allows to see

    def drift_detection(self, y_true, y_pred):
        self.__update(y_true, y_pred)
        if self.pi + self.si >= self.pmin + 3 * self.smin:
            if self.verbose:
                print('Drift detected: time_step={0}'.format(self.ctr))
            return True
        elif self.pi + self.si < self.pmin + 2 * self.smin:
            return False
        elif self.pmin + 2 * self.smin <= self.pi + self.si < self.pmin + 3 * self.smin:
            if self.verbose:
                print('Warning a drift may happens: time_step={0}'.format(self.ctr))
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
