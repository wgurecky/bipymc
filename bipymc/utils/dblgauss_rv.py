#!/usr/bin/python
##
# Description: Implements 2d bimodal gaussian
##
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class BimodeGauss_2D(object):
    def __init__(self, mu_g1=[0, 0], mu_g2=[2, 2], sigma_g1=[0.25, 0.25], sigma_g2=[0.25, 0.25],
                 rho_g1=0.8, rho_g2=-0.8, w_g1=0.25, w_g2=0.75):
        self.mu_g1 = mu_g1
        self.mu_g2 = mu_g2
        cov_g1 = np.array([[sigma_g1[0] ** 2.0, rho_g1 * (sigma_g1[0] * sigma_g1[1])], \
                            [rho_g1 * (sigma_g1[0] * sigma_g1[1]), sigma_g1[1] ** 2.0]])
        cov_g2 = np.array([[sigma_g2[0] ** 2.0, rho_g2 * (sigma_g2[0] * sigma_g2[1])], \
                            [rho_g2 * (sigma_g2[0] * sigma_g2[1]), sigma_g2[1] ** 2.0]])
        self.cov_g1 = cov_g1
        self.cov_g2 = cov_g2
        self.rv_2d_g1 = multivariate_normal(self.mu_g1, self.cov_g1)
        self.rv_2d_g2 = multivariate_normal(self.mu_g2, self.cov_g2)
        self.w_g1 = w_g1 / (w_g1 + w_g2)
        self.w_g2 = w_g2 / (w_g1 + w_g2)

    def pdf(self, y1, y2):
        pos = np.dstack((y1, y2))
        return self.w_g1 * self.rv_2d_g1.pdf(pos) + self.w_g2 * self.rv_2d_g2.pdf(pos)

    def ln_like(self, y):
        assert len(y) == 2
        return np.log(self.pdf(y[0], y[1]))

    def rvs(self, n_samples):
        pdf_select = np.random.choice((True, False), p=(self.w_g1, self.w_g2), size=n_samples)
        samples = np.zeros((n_samples, 2))
        rv_g1_samples = self.rv_2d_g1.rvs(size=n_samples)
        rv_g2_samples = self.rv_2d_g2.rvs(size=n_samples)
        samples[pdf_select, :] = rv_g1_samples[pdf_select, :]
        samples[~pdf_select, :] = rv_g2_samples[~pdf_select, :]
        return (samples[:, 0], samples[:, 1])


if __name__ == "__main__":
    banana = BimodeGauss_2D()
    y1, y2 = banana.rvs(10000)
    plt.figure()
    plt.scatter(y1, y2, s=2, alpha=0.3)
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("banana_plot_samples_ex.png")
    plt.close()

    mean = (np.mean(y1), np.mean(y2))
    print("mean: ", mean)

    plt.figure()
    y1 = np.linspace(-4, 4, 100)
    y2 = np.linspace(-2, 8, 100)
    y1, y2 = np.meshgrid(y1, y2)
    p = banana.pdf(y1, y2)
    plt.contourf(y1, y2, p)
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("banana_plot_pdf_ex.png")
    plt.close()
