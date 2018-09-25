import numpy as np

class DistributionComputer:

    def __init__(self, 
        model_trainer,
        data,
        is_in_disagreement_region,
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
        self.data = data
        self.samples = {i : x for (x, _, _) in self.data}
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
        self.h_erm = h

        self._set_dr()
        self._set_P_min() 
        self._set_I()

        self.dual_vars = None
        self.P = None
        self.max_rounds = 1000

    def get_P(self):

        return self.P

    def compute_P(self):
        
        # TODO: figure out better way to index duals
        dual_vars = {}
        i = 0

        while i < self.max_rounds:
            
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

            i += 1

        self.dual_vars = dual_vars
        self.P = P_lambda

    def _is_converged(self, P_lambda, h, bound):

        func = lambda d: self.I(d, h) / P_lambda(d)
        expectation = self._get_expectation(func)

        return expectation - bound <= self.tolerance

    def _get_expectation(self, get_val):

        vals = [get_val(x) for x in self.samples.values()]

        return sum(vals) / len(vals)

    def _get_dual_update(self, P_lambda, q_lambda, h_bar, bound_bar):

        numer_func = lambda d: self.I(d, h_bar) / P_lambda(d)
        denom_func = lambda d: self.I(d, h_bar) / q_lambda(d)**3
        numer_ex = self._get_expectation(numer_func)
        denom_ex = self._get_expectation(denom_func)

        return 2 * (numer_ex - bound) / denom_ex

    def _get_h_bar(self):
        pass

    def _get_bound(self, h):

        # Compute alpha term
        func = lambda x: self.I(x, h)
        I_ex = self._get_expectation(func)
        alpha_term = 2 * self.alpha**2 * I_ex

        # Compute beta term
        consts = 2 * self.beta**2 * self.tau * self.big_delta * self.gamma
        regret = self.mt.get_error(h, self.data) - self.error_erm
        beta_term = regret * consts

        # Compute xi term
        xi_term = self.xi * self.big_delta**2 * self.tau

        return alpha_term + beta_term + xi_term

    def _get_P_and_q_lambda(self, dual_vars):

        def q_lambda(d):
            pass

        def P_lambda(d):

            p = 0

            if self._in_dr(d):
                ql = q_lambda(d[1])
                p = ql / (ql + 1)

            return p

        return (P_lambda, q_lambda)

    def _set_dr(self):

        self.dr = []

        for (i, x) in self.samples.items():
            if in_is_in_disagreement_region(x):
                self.dr.append(i)

        self._in_dr = lambda d: d[0] in self.dr

    def _set_P_min(self):

        error_erm = self.mt.get_error(self.h_erm, self.data)
        sqrt_term = self.tau * error / (self.epsilon * self.m)
        denom = np.sqrt(sqrt_term) + np.log(self.tau)
        option1 = self.c3 / denom

        self.error_erm = error_erm
        self.P_min = min([option1, 0.5])

    def _set_I(self):

        def I(d, h):
            
            x = d[1]
            input_pred = self.mt.get_prediction(x, h)
            erm_pred = self.mt.get_prediction(x, self.h_erm)
            preds_different = not erm_pred == input_pred

            return self._in_dr(d) and preds_different

        self.I = I
