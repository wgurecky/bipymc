##
# @brief Visualizes bayesian calibration
#
# author:  William Gurecky
# date:    Dec 2018
# modified:  March 2020
##
from __future__ import print_function, division
import numpy as np
import scipy.stats as stats
import mc_plot
import argparse
import h5py


def conv_checkpoint_h5(mcmc_checkpoint_h5='chain_checkpoint_bipymc.h5'):
    h5fr = h5py.File(mcmc_checkpoint_h5, 'r')

    # read size of chains
    tmp_chain = h5fr['chains/chain_id_0'][:]
    n_samples, n_dim = tmp_chain.shape
    tmp_labels = ['theta_%d' % n for n in range(1, n_dim+1)]

    # read all chains from h5 file
    all_chain_list = []
    for i in range(999):
        try:
            tmp_chain = h5fr['chains/chain_id_%d' % i][:]
            all_chain_list.append(tmp_chain)
        except:
            break
    n_chains = i

    # interlace results
    full_chain = np.zeros((n_samples * n_chains, n_dim))
    for j in range(n_chains):
        full_chain[j::n_chains, :] = all_chain_list[j]

    # clean up
    h5fr.close()
    return full_chain, n_chains, tmp_labels


def plot_results(mcmc_out_file='full_chain.txt', num_eval=400,
                 output="mcmc", n_indep_chains=50):
    """
    Plots bayesian calibration results.

    Args:
        mcmc_out_file: string. txt file containing samples from mcmc run.
        Contains table of shape (N, M).  Where M is number of params and
        N is number of samples.
        Each column representing a different calibration var
        and read row is a sample.
        num_eval: int.  Number of samples used to compute statistics
        output: str. Output file name.
        n_indep_chains: int.  Number of chains in the mcmc_out_file.
    """
    # read in chain results
    try:
        mcmc_results = np.loadtxt(mcmc_out_file, skiprows=1, delimiter=', ')
        with open(mcmc_out_file, 'r') as f:
            labels = f.readlines()[0].split(',')

        full_chain = mcmc_results[:, :]
    except:
        full_chain, n_indep_chains, labels = \
            conv_checkpoint_h5(mcmc_out_file)

    # plot the results and compute chain mean
    num_tot = len(full_chain)
    chain = full_chain[-num_eval:, :]
    nburn = num_tot - num_eval

    means = np.mean(chain, axis=0)
    quants = np.percentile(chain, [5, 50, 95], axis=0)
    sdevs = np.std(chain, axis=0)
    result_table = np.zeros((5, means.size))
    print("=== MCMC Results ===")
    print("labels: %s" % str(labels))
    print("mean:  %s" % str(tuple(means)))
    print("+/-1sigma: %s " % str(tuple(sdevs)))
    print("quantiles")
    print("q_5")
    print(str(tuple(quants[0,:])))
    print("q_50")
    print(str(tuple(quants[1,:])))
    print("q_95")
    print(str(tuple(quants[2,:])))
    result_table[0, :] = means
    result_table[1, :] = sdevs
    result_table[2, :] = quants[0,:]
    result_table[3, :] = quants[1,:]
    result_table[4, :] = quants[2,:]
    print("--- Latex table --- ")
    for row in result_table.T:
        for col_val in row:
            print("\\num{ %0.4e } & " % (col_val), end="")
        print("")

    # vis the parameter estimates
    mc_plot.plot_mcmc_params(chain,
            labels=labels,
            savefig='walt_mcmc_plot_' + output + '.png')
    # vis the full chain
    mc_plot.plot_mcmc_chain(full_chain,
            labels=labels,
            savefig='walt_chain_plot_' + output + '.png', nburn=nburn)
    mc_plot.plot_mcmc_indep_chains(full_chain, n_indep_chains,
            labels=labels,
            savefig='walt_chains_plot_' + output + '.png', nburn=nburn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot mcmc results')
    parser.add_argument('-i', type=str, help="MCMC output chains file")
    parser.add_argument('-n', type=int, help="N eval", default=2000)
    parser.add_argument('-o', type=str, help="output string", default="mcmc")
    parser.add_argument('-c', type=int, help="n chains", default=50)
    args = parser.parse_args()
    plot_results(args.i, args.n, args.o, args.c)
