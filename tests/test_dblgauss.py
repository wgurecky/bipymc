#!/usr/bin/python
##
# Description: Tests samplers on a 2d bimodal_gauss shaped distribution
##
from __future__ import print_function, division
import unittest
import numpy as np
from six import iteritems
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
#
from bipymc.utils import banana_rv, dblgauss_rv
from bipymc.mc_plot import mc_plot
from bipymc.demc import DeMcMpi
from bipymc.dream import DreamMpi
from bipymc.dram import DrMetropolis, Dram
np.random.seed(42)


class TestMcmcDblGauss(unittest.TestCase):
    def setUp(self):
        """
        Setup the bimodal_gauss distribution
        """
        self.comm = MPI.COMM_WORLD
        self.bimodal_gauss = dblgauss_rv.BimodeGauss_2D()
        self.sampler_dict = {
            'demc': (self._setup_demc(self.bimodal_gauss.ln_like), 20),
            'dream': (self._setup_dream(self.bimodal_gauss.ln_like), 10),
            'dram': (self._setup_dram(self.bimodal_gauss.ln_like), 1),
            }

        if self.comm.rank == 0:
            # plot true pdf and true samples
            self._plot_bimodal_gauss()

    def test_samplers(self):
        """
        Test ability of each mcmc method to draw samples
        from the 2d bimodal_gauss distribution.
        """
        n_samples = 100000
        n_burn = 40000
        for sampler_name, (my_mcmc, n_chains) in iteritems(self.sampler_dict):
            t0 = time.time()
            try:
                my_mcmc.run_mcmc(n_samples)
            except:
                my_mcmc.run_mcmc(n_samples, [0.0, 0.0])
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
                    self.assertAlmostEqual(1.5, theta_est[0], delta=0.1)
                    self.assertAlmostEqual(1.5, theta_est[1], delta=0.1)
                elif sampler_name == 'dram':
                    self.assertNotAlmostEqual(1.5, theta_est[0], delta=0.1)
                    self.assertNotAlmostEqual(1.5, theta_est[1], delta=0.1)
                else:
                    pass

                # plot mcmc samples
                plt.figure()
                plt.scatter(y1, y2, s=2, alpha=0.10)
                plt.grid(ls='--', alpha=0.5)
                plt.xlim(-2, 4)
                plt.ylim(-2, 4)
                plt.axvline(theta_est[0], -10, 10, c='r')
                plt.axhline(theta_est[1], -10, 10, c='r')
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.title(sampler_name + "  " + r"$ \mu=(%0.2f,%0.2f)$" % (theta_est[0], theta_est[1]))
                plt.savefig(str(sampler_name) + "_bimodal_gauss_sample.png")
                plt.close()

                # plot mcmc chains
                mc_plot.plot_mcmc_indep_chains(full_chain, n_chains,
                        labels=["x1", "x2"],
                        savefig=str(sampler_name) + "_chains.png",
                        scatter=True, nburn=int(n_burn))

    def _plot_bimodal_gauss(self):
        n_samples = 10000
        y1, y2 = self.bimodal_gauss.rvs(n_samples)
        plt.figure()
        plt.scatter(y1, y2, s=2, alpha=0.3)
        plt.xlim(-2, 4)
        plt.ylim(-2, 4)
        plt.grid(ls='--', alpha=0.5)
        plt.axvline(1.5, -10, 10, c='r')
        plt.axhline(1.5, -10, 10, c='r')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("True  " + r"$ \mu=(%0.2f,%0.2f)$" % (1.5, 1.5))
        plt.savefig("true_bimodal_gauss_asamples.png")
        plt.close()

        plt.figure()
        y1 = np.linspace(-2, 4, 100)
        y2 = np.linspace(-2, 4, 100)
        y1, y2 = np.meshgrid(y1, y2)
        p = self.bimodal_gauss.pdf(y1, y2)
        plt.contourf(y1, y2, p)
        plt.grid(ls='--', alpha=0.5)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar(label='probability density')
        plt.savefig("true_bimodal_gauss_pdf.png")
        plt.close()

    def _setup_dram(self, log_like_fn, n_chains=1):
        my_mcmc = Dram(log_like_fn)
        return my_mcmc

    def _setup_demc(self, log_like_fn, n_chains=20):
        theta_0 = [0.0, 0.0]
        my_mcmc = DeMcMpi(log_like_fn, theta_0, n_chains=n_chains, mpi_comm=self.comm)
        return my_mcmc

    def _setup_dream(self, log_like_fn, n_chains=10):
        n_burn = 20000
        theta_0 = [0.0, 0.0]
        my_mcmc = DreamMpi(log_like_fn, theta_0, n_chains=n_chains, mpi_comm=self.comm,
                n_cr_gen=50, burnin_gen=int(n_burn / 10))
        return my_mcmc

