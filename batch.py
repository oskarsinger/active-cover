import numpy as np

class ActiveCover:

    def __init__(
        get_error,
        get_argmin,
        c1, c2, c3, 
        delta, 
        gamma, 
        alpha, beta, xi):

        # Subroutines for specific model
        self.get_error = get_error
        self.get_argmin = get_argmin

        # Non-intuitive parameters
        self.cs = (c1, c2, c3)
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.tau0 = 3

        # Sampling distribution for picking new data points 
        self.P_m = None

        # Which data indexes called queries
        self.oracle_calls = []

        # Current labeled dataset
        self.Z_m = []

        # Current unlabeled dataset
        self.S = []

        # Current hypothesis subset
        self.A_m = None

        # Indicates whether waiting on oracle label
        self.waiting = False

        # Current time step
        self.t = 0

        # Current epoch Number
        self.m = 0

        self._set_epsilon_and_tau()

    def get_label(self, X):

        self.t += 1

        if self.t == self.tau_m + 1:
            self._do_epoch_update()

        in_dr = self._is_in_disagreement_region(X)

        if in_dr:

    def set_label(self, y):
        pass

    def _do_epoch_update(self):

        self._update_h_min()
        self._update_A_m()
        self._update_P_m()

        self.m += 1

    def _update_h_min(self):
        pass

    def _update_A_m(self):

        big_delta_m = self._get_big_delta(self.h_min)

    def _get_big_delta(self, h):

        error = self.get_error(h, self.Z)
        sqrt_term = np.sqrt(self.epsilon_m * error)
        log_term = self.epsilon_m * np.log(self.tau_m)

        return self.cs[0] * sqrt_term + self.cs[1] * log_term

    def _is_in_disagreement_region(self, X):
        pass 

    def _set_epsilon_and_tau(self):

        self.tau_m = self.tau0**(self.m)
        numer = 32 * np.log(
            self.H_size * self.tau_m / self.delta)
        self.epsilon_m = numer / self.tau_m
