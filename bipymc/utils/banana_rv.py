#!/usr/bin/python
##
# Description: Implements 2d banana distribution
##
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class Banana_2D(object):
    def __init__(self, mu1=0, mu2=0, sigma1=1, sigma2=1, rho=0.9, a=1.15, b=0.5):
        self.mu1 = mu1
        self.mu2 = mu2
        # cov params
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        # transform params
        self.a = a
        self.b = b
        # define gauss dist
        mean_vec = np.array([self.mu1, self.mu2])
        cov = np.array([[self.sigma1 ** 2.0, self.rho * (sigma1 * sigma2)], [self.rho * (sigma1 * sigma2), self.sigma2 ** 2.0]])
        self.rv_2d_normal = multivariate_normal(mean_vec, cov)

    def pdf(self, y1, y2):
        # transform coords
        x1_inv = y1 / self.a
        x2_inv = (y2 - self.b * (x1_inv ** 2.0 + self.a ** 2.0)) * self.a

        pos = np.dstack((x1_inv, x2_inv))
        # eval gauss pdf at tranformed coords
        return self.rv_2d_normal.pdf(pos)

    def ln_like(self, y):
        assert len(y) == 2
        return np.log(self.pdf(y[0], y[1]))

    def check_prob_lvl(self, y1, y2, pdf_lvl):
        return pdf_lvl < self.pdf(y1, y2)

    def transform(self, x1, x2):
        y1 = self.a * x1
        y2 = x2 / self.a + self.b * (x1 ** 2.0 + self.a ** 2.0)
        return y1, y2

    def inv_transform(self, y1, y2):
        x1_inv = y1 / self.a
        x2_inv = (y2 - self.b * (x1_inv ** 2.0 + self.a ** 2.0)) * self.a
        return x1_inv, x2_inv

    def rvs(self, n_samples):
        # sample from gauss
        samples = self.rv_2d_normal.rvs(size=n_samples)
        x1_sample, x2_sample = samples[:, 0], samples[:, 1]

        # move sample to transformed coords
        y1 = self.a * x1_sample
        y2 = x2_sample / self.a + self.b * (x1_sample ** 2.0 + self.a ** 2.0)
        return (y1, y2)

    def cdf(self, y1, y2):
        # invert transform coords
        x1_inv = y1 / self.a
        x2_inv = (y2 - self.b * (x1_inv ** 2.0 + self.a ** 2.0)) * self.a
        pos = np.dstack((x1_inv, x2_inv))

        # eval gauss cdf at inv transform coords
        return self.rv_2d_normal.cdf(pos)


if __name__ == "__main__":
    banana = Banana_2D()
    y1, y2 = banana.rvs(10000)
    plt.figure()
    plt.scatter(y1, y2, s=2, alpha=0.3)
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("banana_plot_samples_ex.png")
    plt.close()

    plt.figure()
    y1 = np.linspace(-4, 4, 100)
    y2 = np.linspace(-2, 8, 100)
    y1, y2 = np.meshgrid(y1, y2)
    p = banana.pdf(y1, y2)
    plt.contourf(y1, y2, p)
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("banana_plot_pdf_ex.png")
    plt.close()

    plt.figure()
    y1 = np.linspace(-4, 4, 100)
    y2 = np.linspace(-2, 8, 100)
    y1, y2 = np.meshgrid(y1, y2)
    c = banana.cdf(y1, y2)
    plt.contourf(y1, y2, c)
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("banana_plot_cdf_ex.png")
    plt.close()
