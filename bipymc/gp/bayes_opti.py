#!/usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
# Copyright (c) 2017, William Gurecky
# All rights reserved.
#
# DESCRIPTION: Bayesian optimization using gaussian process regression and
# thompson sampling.  Parallel optimization of an arbitray N dimensional function
# is possible via mpi4py.
#
# OPTIONS: --
# AUTHOR: William Gurecky
# CONTACT: william.gurecky@gmail.com
#==============================================================================
from mpi4py import MPI
import numpy as np
from gpr import *


class bo_optimizer(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self, f, dim, p_bounds=None, x0=None, y0=None, n_init=None,
                 fn_args=[], fn_kwargs={}, comm=MPI.COMM_WORLD):
        self.obj_f = lambda x: f(x, *fn_args, **fn_kwargs)
        # eval obj fn n_init times to seed the method
        self.gp_model = gp_regressor(dim)
        if x0 is not None:
            x_init = x0
            y_init = y0
        else:
            x_init, y_init = self.sample_uniform()
        self.gp_model.fit(x_init, y_init, kwargs.get("y_sigma", 1e-12))

    def optimize(self, n_iter=10):
        # fit GP to current data


        # everyone get a proposal sample
        x_proposal = self.sample_thompson(self.p_bounds_array)

        # collect and distribute proposals to everyone
        pass


    def sample_uniform(self):
        pass

    def sample_thompson(self, sample_bounds, current_depth=0, depth=2):
        # draw sample from gp model
        n = 1000
        sample_grid = []
        for bounds in sample_bounds:
            start, end = bounds[0], bounds[1]
            sample_grid.append(np.linspace(bounds[0], bounds[1], n))
        x_mesh = np.meshgrid(*sample_grid)
        x_mesh_flat = []
        for x_ in x_mesh:
            x_mesh_flat.append(x_.flatten())
        sample_x = np.array(x_mesh_flat).T
        sample_y = self.gp_model.sample_y(sample_x)
        idx_min = np.argmin(sample_y)
        best_x = sample_x[idx_min]
        # TODO: Shrink sample bounds around best_x and recurse
        return best_x


    @property
    def p_bounds_array(self):
        pass




class proposal_generator(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self):
        pass

    def acqui_function(self, *params):
        """!
        @brief Acquisition function abstract method.
            Determines where to select the next sample point.
        """
        pass

    def gen_proposal(self):
        """!
        @brief Returns the next sample point.
        """
        pass


class bo_thompson(bo_optimizer):
    def __init__(self):
        pass
