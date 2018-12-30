import numpy as np

class BatchActiveCover:


    # TODO: put nice defaults here
    def __init__(self,
        model_trainer,
        tolerance,
        c1, c2, c3, 
        delta, 
        gamma, 
        alpha, beta, xi):

        # Subroutines for specific model
        self.mt = model_trainer

        # Non-intuitive parameters
        self.tolerance = tolerance
        self.cs = (c1, c2, c3)
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.xi = xi

        # Epoch base and threshold
        self.tau0 = 3
        self.tau_m = 0

        # Sampling distribution for picking new data points 
        self.P_m = None

        # Which data indexes called queries
        self.oracle_calls = []

        # Labeled data up to most recent epoch
        self.Z_m = []

        # Current epoch dataset
        self.S = []

        # Indicates whether waiting on oracle label
        self.waiting = False

        # Current time step
        self.t = 0

        # Current epoch Number
        self.m = 0

        # Current disagreement region membership
        self.dr = {}

        self._set_epsilon_and_tau()


    def get_parameters(self):

        return self.mt.get_parameters()


    def get_label(self, x):

        label = None

        if self.waiting:
            print('You must provide label for most recent example.')
        else:
            self.t += 1
            new_sample = None

            if self.t == self.tau_m + 1:
                self._do_epoch_update()

            in_dr = self._is_in_disagreement_region(x)

            if in_dr:
                p = self.P_m(x)
                query = np.random.uniform() < p
                new_sample = (x, None, 1 / p) if query else (x, 1, 0)
                self.waiting = query
            else:
                label = self.mt.get_prediction(self.h_erm, x)
                new_sample = (X, label, 1)

            self.S.append(new_sample)

        return label


    def set_label(self, y):

        (x, _, p_inv) = self.S[-1]
        self.S[-1] = (x, y, p_inv)
        self.waiting = False 


    def _do_epoch_update(self):

        self._update_h_erm()
        self._update_P_m()
        self.Z_m.extend(self.S)

        self.m += 1
        self.S = []


    def _update_h_erm(self):

        self.h_erm = self.mt.get_erm(self.Z_m)
        self.big_delta = self._get_big_delta(self.h_erm)


    def _update_P_m(self):

        dc = DistributionComputer(
            model_trainer,
            self.Z_m,
            self._is_in_disagreement_region,
            self.tolerance,
            self.m,
            self.cs[2],
            self.tau_m,
            self.alpha,
            self.beta,
            self.gamma,
            self.epsilon_m,
            self.xi,
            self.big_delta,
            self.h_erm)

        dc.compute_P()

        self.P_m = dc.get_P()


    def _get_big_delta(self):

        error = self.mt.get_error(self.h_erm, self.Z_m)
        sqrt_term = np.sqrt(self.epsilon_m * error)
        log_term = self.epsilon_m * np.log(self.tau_m)

        return self.cs[0] * sqrt_term + self.cs[1] * log_term


    def _is_in_disagreement_region(self, x):
        # TODO: look in App. F of Online Importance Weight Aware Updates to learn about single-constraint optimization with unconstrained oracle to check for disagreement region membership
        threshold = self.delta * self.big_delta


    def _set_epsilon_and_tau(self):

        self.tau_m = self.tau0**(self.m)
        numer = 32 * np.log(
            self.H_size * self.tau_m / self.delta)
        self.epsilon_m = numer / self.tau_m
