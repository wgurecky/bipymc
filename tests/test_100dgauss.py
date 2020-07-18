#!/usr/bin/python
##
# Description: Tests samplers on a 100d gauss shaped distribution
##
from __future__ import print_function, division
import unittest
import numpy as np
from six import iteritems
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
#
from bipymc.utils import banana_rv, dblgauss_rv, d100_gauss
from bipymc.mc_plot import mc_plot
from bipymc.demc import DeMcMpi
from bipymc.dream import DreamMpi
from bipymc.dram import DrMetropolis, Dram
np.random.seed(42)
n_samples = 500000
n_burn = 200000


class TestMcmc100DGauss(unittest.TestCase):
    def setUp(self):
        """
        Setup the 100D gauss distribution
        """
        self.comm = MPI.COMM_WORLD
        self.gauss = d100_gauss.Gauss_100D()
        self.sampler_dict = {
            'demc': (self._setup_demc(self.gauss.ln_like), 200),
            'dream': (self._setup_dream(self.gauss.ln_like), 100),
            }

        if self.comm.rank == 0:
            # plot true pdf and true samples
            self._plot_gauss()

    def test_samplers(self):
        """
        Test ability of each mcmc method to draw samples
        from the 100d gauss distribution.
        """
        global n_burn
        global n_samples
        for sampler_name, (my_mcmc, n_chains) in iteritems(self.sampler_dict):
            t0 = time.time()
            my_mcmc.run_mcmc(n_samples)
            t1 = time.time()
            theta_est, sig_est, chain = my_mcmc.param_est(n_burn=n_burn)
            theta_est_, sig_est_, full_chain = my_mcmc.param_est(n_burn=0)

            if self.comm.rank == 0:
                print("=== " + str(sampler_name) + " ===")
                print("Sampler walltime: %d (s)" % int(t1 - t0))
                print("Esimated params: %s" % str(theta_est))
                print("Estimated params sigma: %s " % str(sig_est))
                print("Acceptance fraction: %f" % my_mcmc.acceptance_fraction)
                try:
                    print("P_cr: %s" % str(my_mcmc.p_cr))
                except:
                    pass
                y1, y2 = chain[:, 0], chain[:, 1]

                if sampler_name == 'demc' or sampler_name == 'dream':
                    self.assertAlmostEqual(0.0, theta_est[0], delta=0.2)
                    self.assertAlmostEqual(0.0, theta_est[1], delta=0.2)

                # plot mcmc samples
                plt.figure()
                plt.scatter(y1, y2, s=2, alpha=0.08)
                plt.grid(ls='--', alpha=0.5)
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.savefig(str(sampler_name) + "_100d_gauss_slice_sample.png")
                plt.close()

                # plot mcmc chains
                """
                mc_plot.plot_mcmc_indep_chains(full_chain, n_chains,
                        labels=["x1", "x2"],
                        savefig=str(sampler_name) + "_chains.png",
                        scatter=True)
                """

    def _plot_gauss(self):
        n_samples = 10000
        y = self.gauss.rvs(n_samples)
        y1, y2 = y[:, 0], y[:, 1]
        plt.figure()
        plt.scatter(y1, y2, s=2, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid(ls='--', alpha=0.5)
        plt.savefig("true_100d_gauss_slice_samples.png")
        plt.close()

    def _setup_demc(self, log_like_fn, n_chains=200):
        theta_0 = np.zeros(100)
        my_mcmc = DeMcMpi(log_like_fn, theta_0, n_chains=n_chains, mpi_comm=self.comm)
        return my_mcmc

    def _setup_dream(self, log_like_fn, n_chains=100):
        global n_burn
        theta_0 = np.zeros(100)
        my_mcmc = DreamMpi(log_like_fn, theta_0, n_chains=n_chains, mpi_comm=self.comm,
                n_cr_gen=50, burnin_gen=int(n_burn / n_chains))
        return my_mcmc

