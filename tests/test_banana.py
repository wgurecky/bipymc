#!/usr/bin/python
##
# Description: Tests samplers on a 2d banana shapeed distribution
##
from __future__ import print_function, division
import unittest
import numpy as np
from six import iteritems
from mpi4py import MPI
#
from bipymc.utils import banana_rv
from bipymc.demc import DeMcMpi
from bipymc.dream import DreamMpi


class TestMcmcBanana(unittest.TestCase):
    def setUp(self):
        """
        Setup the banana distribution
        """
        self.banana = banana_rv.Banana_2D()
        self.sampler_dict = {
            'demc': self._setup_demc(self.banana.ln_like),
            'dream': self._setup_dream(self.banana.ln_like),
            }

    def test_samplers(self):
        """
        Test ability of each mcmc method to draw samples
        from the 2d banana distribution.
        """
        n_samples = 10000
        for sampler_name, my_mcmc in iteritems(self.sampler_dict):
            my_mcmc.run_mcmc(n_samples)

    def _setup_demc(self, log_like_fn):
        comm = MPI.MPI_COMM_WORLD
        theta_0 = [0.0, 0.0]
        my_mcmc = DeMcMpi(log_like_fn, theta_0, n_chains=20, mpi_comm=comm)
        return my_mcmc

    def _setup_dream(self, log_like_fn):
        comm = MPI.MPI_COMM_WORLD
        theta_0 = [0.0, 0.0]
        my_mcmc = DreamMpi(log_like_fn, theta_0, n_chains=6, mpi_comm=comm)
        return my_mcmc

