#
#!/usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
# Copyright (c) 2017, William Gurecky
# All rights reserved.
#
# DESCRIPTION: Gaussian process regression
#
# OPTIONS: --
# AUTHOR: William Gurecky
# CONTACT: william.gurecky@gmail.com
#==============================================================================
import numpy as np
from scipy.optimize import minimize

class gp_kernel(object):
    """!
    @brief Used to generate a covarience matrix
    """
    def __init__(self):
        self._params = []
        self.n_params = None

    def eval(self, a, b, *params):
        raise NotImplementedError

    def __call__(self, x0, x1):
        return self.eval(x0, x1, *self._params)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        assert len(params) == self.n_params
        self._params = params


class squared_exp(gp_kernel):
    """!
    @brief Squared exponential kernel
    """
    def __init__(self):
        self._params = [0.1]
        self.n_params = 1

    def eval(self, a, b, *params):
        if params:
            param = params[0]
        else:
            param = self.params[0]
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1/param) * sqdist)


class squared_exp_noise(gp_kernel):
    """!
    @brief Squared exponential kernel with extra noise parameter.
    """
    def __init__(self):
        self._params = [0.1, 1e-8]
        self.n_params = 2

    def eval(self, a, b, *params):
        n = a.shape[0]
        if params:
            param = params[0]
            sigma_n = params[1]
        else:
            param = self.params[0]
            sigma_n = self.params[1]
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1/param) * sqdist) + sigma_n ** 2.0


class squared_exp_noise_mv(gp_kernel):
    """!
    @brief Squared exponential kernel with extra noise parameter with
        individual length scale parameters for each input dimension.
    """
    def __init__(self, n_dim):
        self._params = np.array(list(np.ones((n_dim))) + [1e-6]) * 1e-1
        self.n_params = n_dim + 1

    def eval(self, a, b, *params):
        n = a.shape[0]
        if params:
            l_param = params[:-1]
            sigma_n = params[-1]
        else:
            l_param = self.params[:-1]
            sigma_n = self.params[-1]
        M = np.array(l_param) ** (-2.0) * np.eye(len(l_param))
        cov_matrix =  1.0 * np.exp(-0.5 * np.dot((a-b).reshape(1, -1).T, np.dot(M, a - b))) \
                + sigma_n ** 2.0
        return cov_matrix

    def rel_var_importance(self):
        return np.abs(self.params[:-1]) / np.sum(np.abs(self.params[:-1]))

class gp_regressor(object):
    """!
    @brief Gaussian process regression.  Has a gp_kernel object
    which describes the covarience (spatial-autocorrelation) between
    known samples.
    Using bayes rule we can update our priors to best match the
    the known sample distribution.
    """
    def __init__(self):
        self.cov_fn = squared_exp_noise()
        self.x_known = np.array([])
        self.y_known = np.array([])

    def fit(self, x, y, params_0=None, method="SLSQP"):
        """
        @brief Fit the kernel's shape params to the known data
        """
        if params_0 is not None:
            pass
        else:
            params_0 = self.cov_fn.params
        self.x_known = x
        self.y_known = y
        neg_log_like_fn = lambda p_list: -1.0 * self.log_like(x, y, p_list)
        res = minimize(neg_log_like_fn, x0=params_0, method=method)
        cov_params = res.x
        print("Fitted Cov Fn Params:", cov_params)
        self.cov_fn.params = cov_params

    def predict(self, x_test):
        """
        @brief Obtain mean estimate at points in x
        @param x_test np_ndarray
        """
        n = len(x_test)
        K = self.cov_fn(self.x_known, self.x_known)
        K_plus_sig = K + np.eye(len(self.x_known)) * 1e-12
        L = np.linalg.cholesky(K_plus_sig)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_known))
        return np.dot(self.cov_fn(self.x_known, x_test).T, alpha)

    def predict_sd(self, x_test):
        """
        @brief Obtain standard deviation estimate at x
        @param x_test np_ndarray
        """
        n = len(x_test)
        K = self.cov_fn(self.x_known, self.x_known)
        K_plus_sig = K + np.eye(len(self.x_known)) * 1e-12
        L = np.linalg.cholesky(K_plus_sig)
        k_s = self.cov_fn(self.x_known, x_test)
        v = np.linalg.solve(L, k_s)
        return self.cov_fn(x_test, x_test).diagonal() - np.dot(v.T, v)

    def log_like(self, X, y, *cov_params):
        """!
        @brief Compute:
        \f[
            ln p(y|X) = - 1/2 y^T (K + \sigma_n^2 I)^-1 y - (1/2)ln(trace(L_K)) - (n/2)ln(2\pi)
            where L_k = cholesky(K + \sigma_n^2 I)
        \f]
        @param y np_1d array of known responses
        @param X np_ndarray of known support points
        @param cov_params list of covarience fn parameters
        """
        n = len(y)
        assert n == len(X)
        K = self.cov_fn.eval(X, X, *cov_params[0])
        K_plus_sig = K + np.eye(n) * 1e-12
        L = np.linalg.cholesky(K_plus_sig)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return -0.5 * np.dot(y.T, alpha) - np.trace(L) - (n / 2.0) * np.log(2 * np.pi)

    def sample_y(self, x_test, n_draws=1):
        """
        @brief Draw a single sample from gaussian process regression at x
        @param x_test np_ndarray
        """
        n = len(x_test)
        K = self.cov_fn(self.x_known, self.x_known)
        K_plus_sig = K + np.eye(len(self.x_known)) * 1e-12
        L = np.linalg.cholesky(K_plus_sig)

        K_ss = self.cov_fn(x_test, x_test)
        K_s = self.cov_fn(self.x_known, x_test)
        Lk = np.linalg.solve(L, K_s)
        # get the mean prediction
        mu = self.predict(x_test)
        # compute scaling factor for unit-gaussian draws
        B = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
        # add tailored gauss noise to mean prediction
        return  mu.reshape(-1,1) + np.dot(B, np.random.normal(size=(n, n_draws)))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # sin test data
    Xtrain = np.random.uniform(-4, 4, 6).reshape(-1,1)
    ytrain = np.sin(Xtrain)

    my_gpr = gp_regressor()
    my_gpr.fit(Xtrain, ytrain)

    n = 50
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    ytest_mean = my_gpr.predict(Xtest)

    ytest_samples = my_gpr.sample_y(Xtest, n_draws=200)

    plt.figure()
    plt.scatter(Xtrain, ytrain, label="train")
    plt.plot(Xtest, ytest_mean, label="mean")
    for y_test in ytest_samples.T:
        plt.plot(Xtest, y_test, lw=0.4, ls='-', c='k', alpha=0.1)
    plt.legend()
    plt.savefig("bo_sin_test.png")
