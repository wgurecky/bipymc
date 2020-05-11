##
# @brief Helps visualize mcmc results
#
# author:  William Gurecky
# date:    Dec 2018
# modified:  March 2020
##
from __future__ import print_function, division
import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats


def plot_mcmc_params(samples, labels, savefig='corner_plot.png', truths=None,
                     show_titles=False, plot_contours=False):
    fig = corner.corner(samples, labels=labels,
            truths=truths, use_math_text=True,
            plot_contours=plot_contours,
            plot_density=False, no_fill_contours=True)

    ndims = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndims, ndims))
    for i in range(ndims):
        ax = axes[i, i]
        ax_data = samples[:, i]
        mean_line(ax, ax_data)
    fig.savefig(savefig)


def mean_line(ax, x, **kwargs):
    # ax = plt.gca()
    mean = np.average(x)
    median = np.median(x)
    quants = np.percentile(x, 100.0 * np.array([0.05, 0.95]))
    sd = np.std(x)
    ax.axvline(mean, ls='--', c='k', alpha=0.5)
    ax.axvline(median, ls=':', c='k', alpha=0.5)
    ax.annotate(r"$q_5$ = {:.3e}".format(quants[0]),
                xy=(0.25, 1.18 + 2*0.08), xycoords=ax.transAxes)
    ax.annotate(r"$q_{50}$" + " = {:.3e} (..)".format(median),
                xy=(0.25, 1.18 + 0.08), xycoords=ax.transAxes)
    ax.annotate(r"$q_{95}$" + " = {:.3e}".format(quants[1]),
                xy=(0.25, 1.18), xycoords=ax.transAxes)
    ax.annotate(r"$\mu$ = {:.3e} (--)".format(mean),
                xy=(0.25, 1.10), xycoords=ax.transAxes)
    ax.annotate(r"$\sigma$ = {:.3e}".format(sd),
                xy=(0.25, 1.02), xycoords=ax.transAxes)


def plot_mcmc_indep_chains(samples, n_chains, labels, savefig, truths=None, nburn=None, scatter=False):
    pl.clf()
    n_params = samples.shape[1]
    fig, axes = pl.subplots(n_params, 1, sharex=True, figsize=(8, 9), squeeze=False)
    for c in range(n_chains):
        chain_samples = samples[c::n_chains]
        for i in range(n_params):
            if scatter:
                axes[i, 0].scatter(range(0, chain_samples[:, i].size),
                        chain_samples[:, i].T, color="k", alpha=0.25, s=2)
            else:
                axes[i, 0].plot(chain_samples[:, i].T, color="k", alpha=0.2, lw=1)

    for i in range(n_params):
        mean = np.mean(samples[:, i][nburn:].T)
        sd = np.std(samples[:, i][nburn:].T)
        axes[i, 0].axhline(mean, ls='--', c='k', label=r"$\mu$=%0.4e"
                                                       "\n"
                                                       r"$\sigma$=%0.4e" % (mean, sd))
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(5))
        if truths is not None:
            axes[i, 0].axhline(truths[i], color="#888888", lw=2)
        if nburn is not None:
            axes[i, 0].axvline(nburn / n_chains, color='k', alpha=0.5, ls=':')
        axes[i, 0].set_ylabel(labels[i])
        axes[i, 0].legend(loc=2)
    fig.tight_layout()
    fig.savefig(savefig)


def plot_mcmc_chain(samples, labels, savefig, truths=None, nburn=None):
    pl.clf()
    # count number of cols in samples
    n_params = samples.shape[1]
    fig, axes = pl.subplots(n_params, 1, sharex=True, figsize=(8, 9), squeeze=False)
    for i in range(n_params):
        axes[i, 0].plot(samples[:, i].T, color="k", alpha=0.6, label="chain")
        mean = np.mean(samples[:, i][nburn:].T)
        sd = np.std(samples[:, i][nburn:].T)
        axes[i, 0].axhline(mean, ls='--', c='k', label=r"$\mu$=%0.4e"
                                                       "\n"
                                                       r"$\sigma$=%0.4e" % (mean, sd))
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(5))
        if truths is not None:
            axes[i, 0].axhline(truths[i], color="#888888", lw=2)
        if nburn is not None:
            axes[i, 0].axvline(nburn, color='k', alpha=0.5, ls=':')
        axes[i, 0].set_ylabel(labels[i])
        axes[i, 0].legend(loc=2)
    fig.tight_layout(h_pad=0.0)
    fig.savefig(savefig)
