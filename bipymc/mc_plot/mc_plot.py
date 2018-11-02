import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


def plot_mcmc_params(samples, labels, savefig='corner_plot.png', truths=None):
    fig = corner.corner(samples, labels=labels,
            truths=truths, use_math_text=True)
    fig.savefig(savefig)


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
