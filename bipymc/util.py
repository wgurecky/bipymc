from __future__ import print_function, division
import numpy as np


def var_ball(varepsilon, dim):
    """!
    @brief Draw single sample from tight gaussian ball
    @param varepsilon  float or 1d_array of len dim
    @param dim  dimension of gaussian ball
    """
    eps = 0.
    if np.all(np.asarray(varepsilon) > 0):
        eps = np.random.multivariate_normal(np.zeros(dim),
                                            np.eye(dim) * np.asarray(varepsilon),
                                            size=1)[0]
    return eps

def var_box(varepsilon, dim):
    """!
    @brief Draw single sample from tight uniform box
    @param varepsilon  float or 1d_array of len dim
    @param dim  dimension of uniform distribution
    """
    eps = 0.
    if np.all(np.asarray(varepsilon) > 0):
        eps = np.random.uniform(low=-np.asarray(varepsilon) * np.ones(dim),
                                high=np.asarray(varepsilon) * np.ones(dim))
    return eps


def gelman_rubin_sweep(x, n_samples_req=100):
    """
    @brief Sweeps over the chain history and returns GR diagnostic as a function
    of chain length for each parameter.
    @param x An array of dimension m x n x k, where m is the number of chains,
             n the number of samples, and k is the dimensionality of the param space.
    @returns rhat_n array with shape (k, n - n_samples_req)
    """
    gr_all = []
    n_samples = x.shape[1]
    if n_samples < n_samples_req:
        raise ValueError(
            'Gelman-Rubin diagnostic sweep requires atleast n_samples_req')
    for n in range(1, n_samples):
        if n >= n_samples_req:
            x_partial = x[:, :n, :]
            n_burn = int(x_partial.shape[1] / 2)
            gr = gelman_rubin_partial(x_partial, n_burn)
            gr_all.append(gr)
    return np.asarray(gr_all).T


def gelman_rubin_partial(x, n_burn=0, return_var=False):
    """!
    @brief Helper function to compute GR diagnostic, discarding
    the first n_burn samples from each chain.
    @param x An array of dimension m x n x k, where m is the number of chains,
             n the number of samples, and k is the dimensionality of the param space.
    @param n_burn number of samples to discard, only the final (n - nburn) samples
           will be used in the computation of the GR diagnostic.
    """
    n_samples = x.shape[1]
    assert n_samples > n_burn
    return gelman_rubin(x[:, n_burn:, :], return_var)


def gelman_rubin(x, return_var=False):
    """!
    @brief Computes estimate of Gelman-Rubin diagnostic.
    @param x An array of dimension m x n x k, where m is the number of chains,
             n the number of samples, and k is the dimensionality of the param space.
    @returns  Rhat array float for dim with len == k

    References
    P. Brooks and A. Gelman. General Methods for Monitoring Convergence of Iterative
    Simulations. Journal of Computational and Graphical Statistics. v7. n4. 1998.
    """
    try:
        # For single parameter chain
        m, n = np.shape(x)
    except ValueError:
        # For iterate over each parameter
        return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n

    if return_var:
        return s2

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return R
