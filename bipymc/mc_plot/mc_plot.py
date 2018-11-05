import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_mcmc_params(samples, labels, savefig='corner_plot.png', truths=None, show_titles=False):
    fig = corner.corner(samples, labels=labels,
            truths=truths, use_math_text=True, show_titles=show_titles,
            title_fmt='0.2e', title_kwargs={'fontsize': 9})

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
    sd = np.std(x)
    ax.axvline(mean, ls='--', c='k', alpha=0.5)
    ax.annotate(r"$\mu$ = {:.3e}".format(mean),
                xy=(0.71, 0.93), xycoords=ax.transAxes)
    ax.annotate(r"$\sigma$ = {:.3e}".format(sd),
                xy=(0.71, 0.835), xycoords=ax.transAxes)


def plot_mcmc_chain(samples, labels, savefig, truths=None):
    pl.clf()
    # count number of cols in samples
    n_params = samples.shape[1]
    fig, axes = pl.subplots(n_params, 1, sharex=True, figsize=(8, 9), squeeze=False)
    for i in range(n_params):
        axes[i, 0].plot(samples[:, i].T, color="k", alpha=0.6)
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(5))
        if truths is not None:
            axes[i, 0].axhline(truths[i], color="#888888", lw=2)
        axes[i, 0].set_ylabel(labels[i])
    fig.tight_layout(h_pad=0.0)
    fig.savefig(savefig)
