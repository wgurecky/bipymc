#!/usr/bin/python
##
# Description: Tests samplers on a 2d banana shaped distribution
##
from __future__ import print_function, division
import unittest
import numpy as np
from six import iteritems
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
#
from bipymc.utils import banana_rv
from bipymc.demc import DeMcMpi
from bipymc.dream import DreamMpi
from bipymc.dram import DrMetropolis, Dram
np.random.seed(42)


class TestMcmcBanana(unittest.TestCase):
    def setUp(self):
        """
        Setup the banana distribution
        """
        self.comm = MPI.COMM_WORLD
        sigma1, sigma2 = 1.0, 1.0
        self.banana = banana_rv.Banana_2D(sigma1=sigma1, sigma2=sigma2)
        self.sampler_dict = {
            'demc': self._setup_demc(self.banana.ln_like),
            'dream': self._setup_dream(self.banana.ln_like),
            'dram': self._setup_dram(self.banana.ln_like),
            }

        if self.comm.rank == 0:
            # plot true pdf and true samples
            self._plot_banana()

    def test_samplers(self):
        """
        Test ability of each mcmc method to draw samples
        from the 2d banana distribution.
        """
        n_samples = 100000
        n_burn = 20000
        for sampler_name, my_mcmc in iteritems(self.sampler_dict):
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
                prob_mask_50 = self.banana.check_prob_lvl(y1, y2, 0.18)
                prob_mask_95 = self.banana.check_prob_lvl(y1, y2, 0.018)
                prob_mask_100 = self.banana.check_prob_lvl(y1, y2, 0.0)
                print("Frac under q_50: %0.5f" % (np.count_nonzero(prob_mask_50) / y1.size))
                print("Frac under q_95: %0.5f" % (np.count_nonzero(prob_mask_95) / y1.size))
                self.assertAlmostEqual(self.frac_samples_in_50, (np.count_nonzero(prob_mask_50) / y1.size), delta=0.05)
                self.assertAlmostEqual(self.frac_samples_in_95, (np.count_nonzero(prob_mask_95) / y1.size), delta=0.05)

                # plot mcmc samples
                plt.figure()
                plt.scatter(y1[prob_mask_100], y2[prob_mask_100], s=2, alpha=0.15)
                plt.scatter(y1[prob_mask_95], y2[prob_mask_95], s=2, alpha=0.15)
                plt.scatter(y1[prob_mask_50], y2[prob_mask_50], s=2, alpha=0.15)
                plt.grid(ls='--', alpha=0.5)
                plt.xlim(-4, 4)
                plt.ylim(-2, 10)
                plt.savefig(str(sampler_name) + "_banana_sample.png")
                plt.close()

    def _plot_banana(self):
        n_samples = 1000000
        y1, y2 = self.banana.rvs(n_samples)
        prob_mask_50 = self.banana.check_prob_lvl(y1, y2, 0.18)
        prob_mask_95 = self.banana.check_prob_lvl(y1, y2, 0.018)
        self.frac_samples_in_50 = np.count_nonzero(prob_mask_50) / n_samples
        self.frac_samples_in_95 = np.count_nonzero(prob_mask_95) / n_samples
        print(self.frac_samples_in_50, self.frac_samples_in_95)
        n_samples = 10000
        y1, y2 = self.banana.rvs(n_samples)
        plt.figure()
        y1
        plt.scatter(y1, y2, s=2, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-2, 10)
        plt.grid(ls='--', alpha=0.5)
        plt.savefig("true_banana_asamples.png")
        plt.close()

        plt.figure()
        y1 = np.linspace(-4, 4, 100)
        y2 = np.linspace(-2, 8, 100)
        y1, y2 = np.meshgrid(y1, y2)
        p = self.banana.pdf(y1, y2)
        plt.contourf(y1, y2, p)
        plt.grid(ls='--', alpha=0.5)
        plt.colorbar(label='probability density')
        plt.savefig("true_banana_pdf.png")
        plt.close()

    def _setup_dram(self, log_like_fn):
        my_mcmc = Dram(log_like_fn)
        return my_mcmc

    def _setup_demc(self, log_like_fn):
        theta_0 = [0.0, 0.0]
        my_mcmc = DeMcMpi(log_like_fn, theta_0, n_chains=20, mpi_comm=self.comm)
        return my_mcmc

    def _setup_dream(self, log_like_fn):
        n_burn = 20000
        theta_0 = [0.0, 0.0]
        my_mcmc = DreamMpi(log_like_fn, theta_0, n_chains=10, mpi_comm=self.comm, n_cr_gen=50, burnin_gen=int(n_burn / 10))
        return my_mcmc

