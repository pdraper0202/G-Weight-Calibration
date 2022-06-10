import copy
import numpy as np
import sys
import warnings


EPS = sys.float_info.min

class Calibrator:
    def __init__(
            self, X, d, t,
            q=None, method=['truncated', 'linear'][0], lower_bound=0, upper_bound=10, max_iter=500, tol=1e-06
        ):
        self._X = X
        self._d = d
        self._t = t
        if q is not None:
            self._q = q
        else:
            self._q = np.repeat(1, d.size)
        self._method = method
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._max_iter = max_iter
        self._tol = tol
        # g-weights of the calibration estimator
        self.g = None

    def calibrate(self):
        X = copy.deepcopy(self._X)
        d = copy.deepcopy(self._d)
        t = copy.deepcopy(self._t)
        q = copy.deepcopy(self._q)

        if np.isnan(X).any() or np.isnan(d).any() or np.isnan(t).any() or np.isnan(q).any():
            raise Exception('Input cannot contain missing values')
        if X.shape[1] != t.size:
            raise Exception('X and t have different dimensions')
        if np.isinf(q).any():
            raise Exception('There are Inf values in the q vector')
        if self._method not in ['truncated', 'linear']:
            raise Exception('The specified method is not defined')
        if self._method == 'linear' and \
            (self._lower_bound is not None or self._upper_bound is not None):
            raise Exception('Bounds not allowed for the linear method')

        _dd = np.repeat(d.reshape(-1, 1), t.size, axis=1)
        _qq = np.repeat(q.reshape(-1, 1), t.size, axis=1)
        _left_matrix = np.linalg.pinv(np.matmul(np.transpose(X * _dd * _qq), X), rcond=EPS)
        _right_matrix = np.transpose(t - np.matmul(np.transpose(d.reshape(-1, 1)), X))
        lamda = np.matmul(_left_matrix, _right_matrix)
        g = 1 + q * np.transpose(np.matmul(X, lamda)).flatten()

        if self._method == 'linear' or np.max(np.abs(lamda)) < EPS:
            pass
        elif self._method == 'truncated':
            if self._lower_bound is not None and self._upper_bound is not None:
                if self._upper_bound <= 1 or self._lower_bound >= 1 or self._lower_bound > self._upper_bound:
                    warnings.warn('The conditions lower_bound < 1 < upper_bound are not fulfilled')
            else:
                raise Exception('Specify the bounds')
            
            X1 = copy.deepcopy(X)
            d1 = copy.deepcopy(d)
            t2 = copy.deepcopy(t)
            q1 = copy.deepcopy(q)
            g1 = copy.deepcopy(g)
            g_rng = np.arange(g.size)
            
            num_iter = 0
            while num_iter < self._max_iter:
                num_iter += 1
                g_inbounds = (g > self._lower_bound) & (g < self._upper_bound)
                g_outbounds = (g < self._lower_bound) | (g > self._upper_bound)
                if any(g_outbounds):
                    g = g.clip(self._lower_bound, self._upper_bound)
                    g_rng_inbounds = g_rng[g_inbounds]
                    g_rng_not_inbounds = g_rng[~g_inbounds]
                    if g_rng.size != 0:
                        t2 = t - np.matmul(np.transpose(g[g_rng_not_inbounds] * d[g_rng_not_inbounds]), X[g_rng_not_inbounds, :])
                        X1 = X[g_rng_inbounds, :]
                        d1 = d[g_rng_inbounds]
                        q1 = q[g_rng_inbounds]
                t1 = np.matmul(np.transpose(d1), X1)
                _d1 = np.repeat(d1.reshape(-1, 1), t1.size, axis=1)
                _q1 = np.repeat(q1.reshape(-1, 1), t1.size, axis=1)
                lamda = np.matmul(
                    np.linalg.pinv(np.matmul(np.transpose(X1 * _d1 * _q1), X1), rcond=EPS),
                    (t2 - t1)
                )
                g1 = 1 + q1 * np.transpose(np.matmul(X1, lamda)).flatten()
                g[g_rng_inbounds] = g1

                _crossprod_left = np.transpose(X)
                _crossprod_right = np.repeat((g * d).reshape(-1, 1), X.shape[1], axis=1)
                tr = np.matmul(_crossprod_left, _crossprod_right)[:, 0].flatten()
                if any(t == 0):
                    expr = np.max(np.abs(tr - t))
                else:   
                    expr = np.max(np.abs(tr - t) / t)
                if (expr < self._tol) and all((g >= self._lower_bound) & (g <= self._upper_bound)):
                    break
            if (num_iter == self._max_iter):
                print(f"No convergence in {self._max_iter} iterations")
                print(f"The bounds for the g-weights are {np.min(g)} and {np.max(g)}")
        else:
            g = None

        self.g = g

    def summary(self):
        pass
