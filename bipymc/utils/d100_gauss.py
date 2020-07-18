#!/usr/bin/python
##
# Description: Implements 100d normal dist
##
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class Gauss_100D(object):
    """
    High dimension multivariate normal from LANL DREAM report
    """
    def __init__(self, rho=0.5, dim=100):
        # rho is pairwise correlation between all rvs
        self.mu = np.zeros(dim)
        self.var = np.sqrt(np.arange(dim) + 1.0)
        self.cov = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    self.cov[i][j] = self.var[i] ** 2.0
                else:
                    self.cov[i][j] = self.var[i] * self.var[j] * rho
        self.dim = dim
        self.rho = rho
        self.rv_100d = multivariate_normal(self.mu, self.cov)

    def pdf(self, y):
        # return self.rv_100d.pdf(y) / 1e-100
        return self.rv_100d.pdf(y)

    def ln_like(self, y):
        assert len(y) == self.dim
        return np.log(self.pdf(y))

    def rvs(self, n_samples):
        rv_samples = self.rv_100d.rvs(size=n_samples)
        return rv_samples


if __name__ == "__main__":
    d100_gauss = Gauss_100D()
    y = d100_gauss.rvs(10)
    print(y.shape)

    p = d100_gauss.pdf(np.zeros(100))
    print(p)
    p = d100_gauss.pdf(np.ones(100))
    print(p)

    ln_p = d100_gauss.ln_like(np.zeros(100))
    print(ln_p)
    ln_p = d100_gauss.ln_like(np.ones(100))
    print(ln_p)
    print(np.diag(d100_gauss.cov))
