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
from __future__ import division, print_function
from numba import jit
import abc
import six
from scipy.linalg import cho_solve, solve_triangular
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipydirect import minimize as ncsu_direct_min


@six.add_metaclass(abc.ABCMeta)
class gp_kernel():
    """!
    @brief Used to generate a covarience matrix
    """
    def __init__(self, ndim, params_0=[]):
        self.ndim = ndim
        self.params = params_0
        self._param_bounds = []

    @abc.abstractmethod
    def eval(self, x1, x2, *params):
        raise NotImplementedError

    def __call__(self, x1, x2):
        return self.eval(x1, x2, *self._params)

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        assert ndim > 0
        self._ndim = ndim

    @property
    @abc.abstractmethod
    def n_params(self):
        raise NotImplementedError

    @property
    def param_bounds(self):
        # all params must be positive
        if self._param_bounds:
            return self._param_bounds
        # default param bounds
        p_bounds = []
        for param in self.params:
            p_bounds.append((0.005, 10.0))
        p_bounds[-1] = (0.001, 5e3)
        return p_bounds

    @param_bounds.setter
    def param_bounds(self, p_bounds):
        if len(p_bounds) == len(self._params):
            assert len(p_bounds[0]) == 2
            self._param_bounds = p_bounds
        else:
            raise RuntimeError

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
        params_0 = [1.0]
        super(squared_exp, self).__init__(ndim, params_0)

    def eval(self, x1, x2, *params):
        if params:
            param = params[0]
        else:
            param = self.params[0]
        sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
        return np.exp(-.5 * (1/param) * sqdist)

    @property
    def n_params(self):
        return 1


class squared_exp_noise(gp_kernel):
    """!
    @brief Squared exponential kernel with extra noise parameter.
    """
    def __init__(self, ndim=1):
        params_0 = [1.0, 1.0]
        super(squared_exp_noise, self).__init__(ndim, params_0)

    def eval(self, x1, x2, *params):
        n = x1.shape[0]
        if params:
            param = params[0]
            sigma_n = params[1]
        else:
            param = self.params[0]
            sigma_n = self.params[1]
        sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
        cov_m = np.exp(-.5 * (1/param) * sqdist)
        return cov_m * (sigma_n ** 2.0)

    @property
    def n_params(self):
        return 2


class squared_exp_noise_mv(gp_kernel):
    """!
    @brief Squared exponential kernel with extra noise parameter with
        individual length scale parameters for each input dimension.
    """
    def __init__(self, ndim=1, params_0=None, params_bounds=None):
        params_0 = np.array(list(np.ones((ndim)) * 0.2) + [1.0]) * 1e0
        super(squared_exp_noise_mv, self).__init__(ndim, params_0)

    def eval(self, x1, x2, *params):
        n = x1.shape[0]
        if params:
            l_param = params[:-1]
            sigma_n = params[-1]
        else:
            l_param = self.params[:-1]
            sigma_n = self.params[-1]
        M = np.array(l_param) ** (-2.0) * np.eye(len(l_param))
        cov_m = build_cov(x1, x2, M)
        return cov_m * (sigma_n ** 2.0)

    def rel_var_importance(self):
        return np.abs(self.params[:-1]) / np.sum(np.abs(self.params[:-1]))

    @property
    def n_params(self):
        return self.ndim + 1


@jit(nopython=True)
def build_cov(x1, x2, M):
    cov_m = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            v_diff = x1[i] - x2[j]
            cov_m[i, j] = np.exp(-0.5 * np.dot(v_diff, np.dot(M, (v_diff).T)))
    return cov_m


class gp_regressor(object):
    """!
    @brief Gaussian process regression.  Has a gp_kernel object
    which describes the covarience (spatial-autocorrelation) between
    known samples.
    Using bayes rule we can update our priors to best match the
    the known sample distribution.
    """
    def __init__(self, ndim=1, domain_bounds=None, **kwargs):
        self.verbose = kwargs.get("verbose", True)
        self.cov_fn = squared_exp_noise_mv(ndim)
        self.def_scale(domain_bounds)
        self._x_known = np.array([])
        self.y_known = np.array([])
        # cov matrix storage to prevent unnecisasry recalc of cov matrix
        self.K, self.L, self.alpha = None, None, None
        self.prior = None

    def __call__(self, x_test):
        """
        @brief Obtain mean estimate at points in x
        @param x_test np_ndarray
        """
        return self.predict(x_test)

    def def_scale(self, domain_bounds):
        """!
        @breif Supply scale for each dimension
        """
        self.domain_bounds = domain_bounds
        if domain_bounds is not None:
            assert len(domain_bounds) == self.cov_fn.ndim

    def _x_transform(self, x):
        if self.domain_bounds is not None:
            domain_bounds_min = np.asarray(self.domain_bounds)[:, 0]
            domain_bounds_max = np.asarray(self.domain_bounds)[:, 1]
            x_trans = (x - domain_bounds_min) / (domain_bounds_max - domain_bounds_min)
            return x_trans
        else:
            if self.verbose: print("WARNING: No bounds specified for GP")
            return x

    def x_tr(self, x):
        """!
        @brief Alias of _x_transform
        """
        return self._x_transform(x)

    @property
    def x_known(self):
        return self._x_transform(self._x_known)

    @x_known.setter
    def x_known(self, x_known):
        self._x_known = x_known

    def fit(self, x, y, y_sigma=1e-10, params_0=None, method="direct", **kwargs):
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
        if kwargs.get("y_shift", True):
            self.y_shift = np.mean(y)
        else:
            self.y_shift = 0.0
        self.y_known = y - self.y_shift
        self.y_known_sigma = y_sigma
        if isinstance(self.prior, gp_regressor):
            neg_log_like_fn = lambda p_list: -1.0 * self.log_like(self.x_known, self.y_known, p_list) \
                                             -1.0 * self.prior.log_like(self.prior.x_known, self.prior.y_known, p_list)
        else:
            neg_log_like_fn = lambda p_list: -1.0 * self.log_like(self.x_known, self.y_known, p_list)
        if method == 'direct' or method == 'ncsu':
            res = ncsu_direct_min(neg_log_like_fn, bounds=self.cov_fn.param_bounds,
                                  maxf=kwargs.get('maxf', 700), algmethod=kwargs.get('algmethod', 1),
                                  eps=kwargs.get("eps", 1e-4))
        else:
            _neg_log_like_fn = lambda p_list: neg_log_like_fn(p_list)[0]
            res = basinhopping(_neg_log_like_fn, x0=params_0, T=kwargs.get("T", 5.0), niter_success=12,
                               niter=kwargs.get("niter", 90), interval=10, stepsize=0.1,
                               minimizer_kwargs={'bounds': self.cov_fn.param_bounds, 'method': method})
        cov_params = res.x
        print("Fitted Cov Fn Params:", cov_params)
        self.cov_fn.params = cov_params
        # pre-compute cholosky decomp of cov matrix
        self._update_cholesky_k()

    def set_prior(self, prior):
        assert isinstance(prior, gp_regressor)
        self.prior = prior

    def _update_cholesky_k(self):
        self.K = self.cov_fn(self.x_known, self.x_known) + self.y_known_sigma*np.eye(len(self.x_known))
        K_plus_sig = self.K + np.eye(len(self.x_known)) * 1e-12
        self.L = np.linalg.cholesky(nearestPD(K_plus_sig))
        self.alpha = cho_solve((self.L, True), self.y_known)

    def predict(self, x_test):
        """
        @brief Obtain mean estimate at points in x
        @param x_test np_ndarray
        """
        assert self.alpha is not None
        x_test_tr = self.x_tr(x_test)
        return np.dot(self.cov_fn(self.x_known, x_test_tr).T, self.alpha) + self.y_shift

    def predict_sd(self, x_tests, cov=False, chunk_size=10, **kwargs):
        """
        @brief Obtain standard deviation estimate at x
        @param x_test np_ndarray
        """
        assert self.K is not None
        K, L = self.K, self.L

        res = []
        # chunk problem in to smaller problems
        for x_test in np.array_split(x_tests, int(np.ceil(len(x_tests) / chunk_size)), axis=0):
            x_test = self.x_tr(x_test)
            k_s = self.cov_fn(self.x_known, x_test)
            v = solve_triangular(L, k_s, check_finite=False, lower=True)
            cov_m = self.cov_fn(x_test, x_test) - np.dot(v.T, v)
            res.append(np.sqrt(cov_m.diagonal()))
        return np.array(res).flatten()

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
        K = self.cov_fn.eval(X, X, *cov_params[0]) + self.y_known_sigma*np.eye(len(self.x_known))
        K_plus_sig = K + np.eye(n) * 1e-12
        L = np.linalg.cholesky(nearestPD(K_plus_sig))
        # alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        alpha = cho_solve((L, True), y)
        # note: trace is sum of diag
        return -0.5 * np.dot(y.T, alpha) - np.trace(L) - (n / 2.0) * np.log(2 * np.pi)

    def sample_y(self, x_tests, n_draws=1, diag_scale=1e-6, chunk_size=10, rnd_seed=None):
        """
        @brief Draw a single sample from gaussian process regression at locations x_tests
        @param x_tests np_ndarray  locations at which to sample the gaussian process
        """
        assert self.K is not None
        K, L = self.K, self.L

        if rnd_seed:
            np.random.seed(rnd_seed)

        res = []
        for x_test in np.array_split(x_tests, int(np.ceil(len(x_tests) / chunk_size)), axis=0):
            tr_x_test = self.x_tr(x_test)
            K_ss = self.cov_fn(tr_x_test, tr_x_test)
            K_s = self.cov_fn(self.x_known, tr_x_test)
            Lk = np.linalg.solve(L, K_s)
            # get the mean prediction
            mu = self.predict(x_test)
            # compute scaling factor for unit-gaussian draws
            n = len(tr_x_test)
            B = np.linalg.cholesky(nearestPD(K_ss + diag_scale*np.eye(n) - np.dot(Lk.T, Lk)))
            # add tailored gauss noise to mean prediction
            res.append(mu.reshape(-1,1) + np.dot(B, np.random.normal(size=(n, n_draws))))
        result = np.array(res).reshape(-1, n_draws)
        # expected shape: (len(xtest), n_draws)
        return result

    def sample_y_rnd(self, x_tests, n_draws=1, chunk_size=10, rnd_seed=None, **kwargs):
        """!
        @brief Generate spatially uncorrelated samples from the gaussian process.
        Significantly faster than sample_y() with a large chunk_size.
        """
        if rnd_seed:
            np.random.seed(rnd_seed)

        results = []
        for x_test in np.array_split(x_tests, int(np.ceil(len(x_tests) / chunk_size)), axis=0):
            mu = self.predict(x_test)
            sd = self.predict_sd(x_test, chunk_size=10)
            res = np.random.normal(loc=mu.flatten(), scale=sd.flatten(), size=(n_draws, len(x_test))).T
            results.append(res)
        # expected shape: (len(xtest), n_draws)
        return np.array(results).reshape(-1, n_draws)

    @property
    def is_fit(self):
        """
        @brief Check if model has been fit
        """
        return self.K is not None


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
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
    # sin test data
    Xtrain = np.random.uniform(-4, 4, 20).reshape(-1,1)
    ytrain = np.sin(Xtrain) + np.random.uniform(-8e-2, 8e-2, Xtrain.size).reshape(Xtrain.shape)

    my_gpr = gp_regressor(domain_bounds=((-4., 4.),))
    # my_gpr = gp_regressor(domain_bounds=True)
    my_gpr.fit(Xtrain, ytrain, y_sigma=1e-2)

    n = 500
    Xtest = np.linspace(np.min(Xtrain), np.max(Xtrain), n).reshape(-1,1)
    ytest_mean = my_gpr.predict(Xtest)

    ytest_samples = my_gpr.sample_y(Xtest, n_draws=200, chunk_size=1e10)

    ytest_sd = my_gpr.predict_sd(Xtest)

    plt.figure()
    plt.plot(Xtest, ytest_mean, label="mean")
    for y_test in ytest_samples.T:
        plt.plot(Xtest, y_test, lw=0.1, ls='-', c='k', alpha=0.45)
    plt.plot(Xtest, ytest_mean.flatten() + 3.0 * ytest_sd, c='r', label=r"$\pm3\sigma$")
    plt.plot(Xtest, ytest_mean.flatten() - 3.0 * ytest_sd, c='r')
    plt.scatter(Xtrain, ytrain, label="train")
    plt.grid(axis='both', ls='--', alpha=0.5)
    plt.legend()
    plt.savefig("gp_sin_test.png")
    plt.close()

    # check agains sklearn
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3))
    sk_gpr_1d = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200)
    sk_gpr_1d.fit(Xtrain, ytrain)
    ytest_mean_sk, ytest_sd_sk = sk_gpr_1d.predict(Xtest, return_std=True)
    plt.figure()
    plt.plot(Xtest, ytest_mean_sk, label="mean")
    plt.plot(Xtest, ytest_mean_sk.flatten() + 3.0 * ytest_sd_sk, c='r', label=r"$\pm3\sigma$")
    plt.plot(Xtest, ytest_mean_sk.flatten() - 3.0 * ytest_sd_sk, c='r')
    plt.scatter(Xtrain, ytrain, label="train")
    plt.grid(axis='both', ls='--', alpha=0.5)
    plt.legend()
    plt.savefig("gp_sin_test_sk.png")
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
    my_gpr_nd = gp_regressor(ndim=2, domain_bounds=((-4., 4.), (-4., 4.)))
    my_gpr_nd.fit(Xtrain, ytrain)

    n = 100
    Xtest = np.linspace(np.min(Xtrain[:, 0]), np.max(Xtrain[:, 0]), n)
    Ytest = np.linspace(np.min(Xtrain[:, 1]), np.max(Xtrain[:, 1]), n)
    xt, yt = np.meshgrid(Xtest, Ytest)
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
    my_gpr_2d.fit(Xtrain, Z, y_sigma=1e-2)

    n = 50
    Xtest = np.linspace(-4.0, 4.0, n)
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
    sk_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200)
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
