from __future__ import division, print_function
import numpy as np
import unittest
from bipymc.gp.gpr import gp_regressor
from bipymc.gp.bayes_opti import bo_optimizer
from mpi4py import MPI
np.random.seed(42)


class TestBayesOpt(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_1d_bo(self):
        comm = MPI.COMM_WORLD
        def obj_fn_sin(Xtest):
            return np.sin(Xtest)

        # bounds on params (x, y)
        my_bounds = ((-4, 4),)
        my_bo = bo_optimizer(obj_fn_sin, dim=1, s_bounds=my_bounds, n_init=2, comm=comm)

        # run optimizer
        x_opt, y_opt = my_bo.optimize(20, return_y=True)

        # fmin
        self.assertAlmostEqual(y_opt, -1.0, delta=1e-2)
        self.assertAlmostEqual(x_opt, -np.pi/2., delta=1e-2)
