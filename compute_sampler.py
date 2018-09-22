import numpy as np

class SamplerComputer:

    def __init__(self, 
        model_trainer,
        Z,
        tolerance,
        m,
        c3,
        tau,
        alpha,
        beta,
        gamma,
        epsilon,
        xi,
        big_delta,
        h):

        self.mt = model_trainer
        self.Z = Z
        self.tolerance = tolerance
        self.m = m
        self.c3 = c3
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi 
        self.big_delta = big_delta,
        self.h = h

        self._set_P_min() 

        self.dual_vars = None

    def get_sampler(self):

        # TODO: make sampler from dual vars
        sampler = None

        return sampler

    def compute_sampler(self):
        
        # TODO: figure out dim of duals
        dual_vars = {}

        while True:
            
            (P_lambda, q_lambda) = self._get_P_and_q_lambda(dual_vars)
            h_bar = self._get_h_bar(P_lambda)
            bound_bar = self._get_bound(h_bar)
            converged = self._is_converged(
                P_lambda,
                h_bar,
                bound_bar)

            if converged:
                break

            update = self._get_dual_update(
                P_lambda, 
                q_lambda,
                h_bar,
                bound_bar)

            if h_bar in dual_vars:
                dual_vars[h_bar] += update
            else:
                dual_vars[h_bar] = update

        self.dual_vars = dual_vars


    def _is_converged(self, P_lambda, h, bound):

        # TODO: what is that indicator in the numerator?
        func = lambda x: I(x, h) / P_lambda(x)
        expectation = self._get_expectation(func)

        return expectation - bound <= self.tolerance

    def _get_expectation(self, get_val):

        vals = [get_val(x) for x in self.Z]

        return sum(vals) / len(vals)

    def _get_dual_update(self, P_lambda, q_lambda, h_bar, bound_bar):

        numer_func = lambda x: I(x, h_bar) / P_lambda(x)
        denom_func = lambda x: I(x, h_bar) / q_lambda(x)**3
        numer_ex = self._get_expectation(numer_func)
        denom_ex = self._get_expectation(denom_func)

        return 2 * (numer_ex - bound) / denom_ex

    def _get_h_bar(self):
        pass

    def _get_bound(self, h):
        pass

    def _get_P_and_q_lambda(self, dual_vars):
        pass

    def _set_P_min(self):

        error = self.mt.get_error(self.h, self.Z)
        sqrt_term = self.tau * error / (self.epsilon * self.m)
        denom = np.sqrt(sqrt_term) + np.log(self.tau)
        option1 = self.c3 / denom

        self.P_min min([option1, 0.5])
