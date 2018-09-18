import numpy as np

class ActiveCover:

    def __init__(
        get_error,
        get_argmin,
        c1, c2, c3, 
        delta, 
        gamma, 
        alpha, beta, xi, 
        tau0=2):

        self.get_error = get_error
        self.get_argmin = get_argmin
        self.cs = (c1, c2, c3)
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.tau0 = tau0

        self.sampling_distribution = None
        self.oracle_calls = []
        self.waiting = True
        self.num_rounds = 0
        self.num_epochs = 0
        self.epsilon = 0

    def get_label(self, X):

        self.num_rounds += 1

        if self.num_rounds == self.tau0**(self.num_epochs+1):
            self.num_epochs += 1
            self._retrain()

        if

    def _retrain(self):

        h_min = self.get_argmin(self.oracle_calls)

    def _is_in_disagreement_region(self, X):

        pass
