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
from numba import jit
import numpy as np
from scipy.optimize import minimize

class gp_kernel(object):
    """!
    @brief Used to generate a covarience matrix
    """
    def __init__(self, ndim):
        self._ndim = ndim
        self._params = []
        self.n_params = None

    def eval(self, a, b, *params):
        raise NotImplementedError

    def __call__(self, x0, x1):
        return self.eval(x0, x1, *self._params)

    @property
    def param_bounds(self):
        # defaults to all params must be positive
        p_bounds = []
        for param in self.params:
            p_bounds.append((0.01, 1e10))
        return p_bounds

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
    def __init__(self, ndim=1):
        self._params = [1.0]
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
    def __init__(self, ndim=1):
        self._params = [1.0, 1.0]
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
        cov_m = np.exp(-.5 * (1/param) * sqdist)
        return cov_m * (sigma_n ** 2.0)


class squared_exp_noise_mv(gp_kernel):
    """!
    @brief Squared exponential kernel with extra noise parameter with
        individual length scale parameters for each input dimension.
    """
    def __init__(self, n_dim=1):
        self.ndim = n_dim
        self._params = np.array(list(np.ones((n_dim))) + [1.0]) * 1e0
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
        cov_m = build_cov(a, b, M)
        return cov_m * (sigma_n ** 2.0)

    def rel_var_importance(self):
        return np.abs(self.params[:-1]) / np.sum(np.abs(self.params[:-1]))



@jit(nopython=True)
def build_cov(a, b, M):
    cov_m = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            cov_m[i, j] = np.exp(-0.5 * np.dot(a[i]-b[j], np.dot(M, (a[i] - b[j]).T)))
    return cov_m


class gp_regressor(object):
    """!
    @brief Gaussian process regression.  Has a gp_kernel object
    which describes the covarience (spatial-autocorrelation) between
    known samples.
    Using bayes rule we can update our priors to best match the
    the known sample distribution.
    """
    def __init__(self, ndim=1):
        self.cov_fn = squared_exp_noise_mv(ndim)
        self.x_known = np.array([])
        self.y_known = np.array([])
        # cov matrix storage to prevent unnecisasry recalc of cov matrix
        self.K, self.L, self.alpha = None, None, None

    def fit(self, x, y, y_sigma=1e-10, params_0=None, method="TNC"):
        """
        @brief Fit the kernel's shape params to the known data
        @param x np_ndarray
        @param y np_1darray
        @param y_sigma np_1darray or float
        """
        if params_0 is not None:
            assert isinstance(params_0, list)
        else:
            params_0 = self.cov_fn.params
        self.x_known = x
        self.y_known = y
        self.y_known_sigma = y_sigma
        neg_log_like_fn = lambda p_list: -1.0 * self.log_like(x, y, p_list)
        res = minimize(neg_log_like_fn, x0=params_0, bounds=self.cov_fn.param_bounds, method=method)
        cov_params = res.x
        print("Fitted Cov Fn Params:", cov_params)
        self.cov_fn.params = cov_params
        # pre-compute cholosky decomp of cov matrix
        self._update_cholesky_k()

    def _update_cholesky_k(self):
        self.K = self.cov_fn(self.x_known, self.x_known) + self.y_known_sigma
        K_plus_sig = self.K + np.eye(len(self.x_known)) * 1e-12
        self.L = np.linalg.cholesky(K_plus_sig)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_known))

    def predict(self, x_test):
        """
        @brief Obtain mean estimate at points in x
        @param x_test np_ndarray
        """
        assert self.alpha is not None
        return np.dot(self.cov_fn(self.x_known, x_test).T, self.alpha)

    def predict_sd(self, x_test):
        """
        @brief Obtain standard deviation estimate at x
        @param x_test np_ndarray
        """
        assert self.K is not None
        K, L = self.K, self.L

        k_s = self.cov_fn(self.x_known, x_test)
        v = np.linalg.solve(L, k_s)
        cov_m = self.cov_fn(x_test, x_test) - np.dot(v.T, v)
        return np.sqrt(cov_m.diagonal())

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
        K_plus_sig = K + np.eye(n) * 1e-10
        L = np.linalg.cholesky(nearestPD(K_plus_sig))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        # note: trace is sum of diag
        return -0.5 * np.dot(y.T, alpha) - np.trace(L) - (n / 2.0) * np.log(2 * np.pi)

    def sample_y(self, x_test, n_draws=1, diag_scale=1e-6):
        """
        @brief Draw a single sample from gaussian process regression at x
        @param x_test np_ndarray
        """
        assert self.K is not None
        K, L = self.K, self.L

        K_ss = self.cov_fn(x_test, x_test)
        K_s = self.cov_fn(self.x_known, x_test)
        Lk = np.linalg.solve(L, K_s)
        # get the mean prediction
        mu = self.predict(x_test)
        # compute scaling factor for unit-gaussian draws
        n = len(x_test)
        B = np.linalg.cholesky(K_ss + diag_scale*np.eye(n) - np.dot(Lk.T, Lk))
        # add tailored gauss noise to mean prediction
        return  mu.reshape(-1,1) + np.dot(B, np.random.normal(size=(n, n_draws)))


def nearestPD(A):
    """!
    @brief Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    """!
    @brief Returns true when input is positive-definite, via Cholesky
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # sin test data
    Xtrain = np.random.uniform(-4, 4, 10).reshape(-1,1)
    ytrain = np.sin(Xtrain)

    my_gpr = gp_regressor()
    my_gpr.fit(Xtrain, ytrain)

    n = 50
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    ytest_mean = my_gpr.predict(Xtest)

    ytest_samples = my_gpr.sample_y(Xtest, n_draws=200)

    ytest_sd = my_gpr.predict_sd(Xtest)

    plt.figure()
    plt.plot(Xtest, ytest_mean, label="mean")
    for y_test in ytest_samples.T:
        plt.plot(Xtest, y_test, lw=0.1, ls='-', c='k', alpha=0.7)
    plt.plot(Xtest, ytest_mean.flatten() + 3.0 * ytest_sd, c='r', label=r"$\pm3\sigma$")
    plt.plot(Xtest, ytest_mean.flatten() - 3.0 * ytest_sd, c='r')
    plt.scatter(Xtrain, ytrain, label="train")
    plt.grid(axis='both', ls='--', alpha=0.5)
    plt.legend()
    plt.savefig("gp_sin_test.png")
    plt.close()

    # 2d test
    from scipy.stats import multivariate_normal
    x1 = np.random.uniform(-4, 4, 10)
    y1 = np.random.uniform(-4, 4, 10)
    X, Y = np.meshgrid(x1, y1)
    mu1, cov1 = [0, 0], [[1.0, 0],[0, 1.0]]
    rv1 = multivariate_normal(mu1, cov1)
    Z1 = rv1.pdf(np.dstack((X, Y)))
    mu2, cov2 = [1, 1], [[1.5, 0],[0, 0.5]]
    rv = multivariate_normal(mu2, cov2)
    Z2 = rv.pdf(np.dstack((X, Y)))
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)

    Xtrain = np.array((X.flatten(), Y.flatten())).T
    ytrain = Z.flatten()
    my_gpr_nd = gp_regressor(ndim=2)
    my_gpr_nd.fit(Xtrain, ytrain)

    n = 50
    Xtest = np.linspace(-5, 5, n)
    xt, yt = np.meshgrid(Xtest, Xtest)
    Xtest = np.array((xt.flatten(), yt.flatten())).T
    ytest_mean = my_gpr_nd.predict(Xtest)
    zt = ytest_mean.reshape(xt.shape)

    # plot contour
    contour_plot = plt.figure()
    x_grid, y_grid, z_grid = xt, yt, zt
    plt.subplot(1, 1, 1)
    nlevels = 20
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, nlevels, colors='k', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.scatter(X, Y, c='k', s=3, alpha=0.6)
    contour_plot.savefig("gp_2d_test.png")
    plt.close()

    # 2d quadradic gp fit example
    def obj_fn_2d(Xtest):
        X, Y = Xtest[:, 0], Xtest[:, 1]
        return X ** 2.0 + Y ** 2.0

    x1 = np.random.uniform(-4, 4, 70)
    y1 = np.random.uniform(-4, 4, 70)
    Xtrain = np.array([x1, y1]).T
    Z = obj_fn_2d(Xtrain)

    my_gpr_2d = gp_regressor(ndim=2)
    import pdb; pdb.set_trace()
    my_gpr_2d.fit(Xtrain, Z, y_sigma=1e-2)

    n = 50
    Xtest = np.linspace(-5, 5, n)
    xt, yt = np.meshgrid(Xtest, Xtest)
    Xtest = np.array((xt.flatten(), yt.flatten())).T
    ytest_mean = my_gpr_2d.predict(Xtest)
    zt = ytest_mean.reshape(xt.shape)

    # plot contour
    contour_plot = plt.figure()
    x_grid, y_grid, z_grid = xt, yt, zt
    plt.subplot(1, 1, 1)
    nlevels = 20
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, nlevels, colors='k', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.scatter(x1, y1, c='k', s=3, alpha=0.6)
    contour_plot.savefig("gp_2d_test_quad.png")
    plt.close()

    # sk-learn compare
    from sklearn.gaussian_process import GaussianProcessRegressor
    sk_gpr = GaussianProcessRegressor()
    sk_gpr.fit(Xtrain, Z)
    zt = sk_gpr.predict(Xtest).reshape(xt.shape)

    # plot contour
    contour_plot = plt.figure()
    x_grid, y_grid, z_grid = xt, yt, zt
    plt.subplot(1, 1, 1)
    nlevels = 20
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, nlevels, colors='k', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.scatter(x1, y1, c='k', s=3, alpha=0.6)
    contour_plot.savefig("gp_2d_test_quad_sk.png")
    plt.close()
