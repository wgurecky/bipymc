#!/usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
# Copyright (c) 2017, William Gurecky
# All rights reserved.
#
# DESCRIPTION: Global optimization using Random Embedding Bayesian Optimization
# ref:
# Z. Wang,  F. Hutter, M. Zoghi.  Bayesian optimization in a Billion Dimensions
# via Random Embeddings.  Journal of Artificial Intelligence Research. vol 55. 2016.
#
# OPTIONS: --
# AUTHOR: William Gurecky
# CONTACT: william.gurecky@gmail.com
#==============================================================================
from mpi4py import MPI
import numpy as np
from bayes_opti import *


class rembo_optimizer(bo_optimizer):
    def __init__(self, f, data_dim, embedded_dim, s_bounds, x0=None, y0=None, n_init=2,
                 fn_args=[], fn_kwargs={}, comm=comm, **kwargs):
        """!
        @param n_trials  Number of random embeddings to try in parallel
        """
        # split comm into sub comms for each individual optimizer
        self.original_bounds = s_bounds
        self.d_e = embedded_dim
        self.A_proj = self._gen_random_projection()
        super(rembo_optimizer, self).__init__(f, self.d_e, self.search_bounds, n_init=2,
                 fn_args=[], fn_kwargs={}, comm=comm, **kwargs)
        self.obj_f = lambda x_embedded: self._obj_f(f, x_embedded, *fn_args, **fn_kwargs)

    def _x_transform(self, x):
        """!
        @brief Always transform data to be in domain [-1, 1]^data_dim
        """
        if self.original_bounds is not None:
            domain_bounds_min = np.asarray(self.original_bounds)[:, 0]
            domain_bounds_max = np.asarray(self.original_bounds)[:, 1]
            return 2.0 * ((x - domain_bounds_min) / (domain_bounds_max - domain_bounds_min)) - 1.0
        else:
            raise RuntimeError("ERROR: Must supply search bounds.")

    def _obj_f(self, f, x_embedded, *fn_args, **fn_kwargs):
        x_out = []
        for x_vec in x_embedded:
            x_out = np.dot(self.A_proj, x_vec)
        return f(np.asarray(x_out), *fn_args, **fn_kwargs)

    @property
    def x_known(self):
        return self._x_transform(self._x_known)

    @x_known.setter
    def x_known(self, x_known):
        self._x_known = x_known

    @property
    def d_e(self):
        return self._d_e

    @d_e.setter
    def d_e(self, dim):
        assert dim > 0
        self._d_e = d_e

    def _gen_random_projection(self):
        """!
        @brief Generate a random matrix which maps a vector from data_dim to
        embedded_dim
        """
        original_dim = self.data_dim
        dest_dim = self.d_e
        A_proj = np.zeros((dest_dim, original_dim))
        for i in range(dest_dim):
            for j in range(original_dim):
                A_proj[i, j] = np.random.normal(loc=0.0, scale=1.0, size=1)
        return A_proj

    @property
    def search_bounds(self):
        """!
        @brief Overloads search_bounds prop from bo_optimizer
        Alias for embedded_search_bounds
        """
        return self.embedded_search_bounds()

    def embedded_search_bounds(self):
        """!
        @brief convert original bounds to bounds in embedded_dim.  Ex: convert
        bounds of a R^3 space to bounds in an R^2 space.
        """
        d_e_search_bounds = []
        for i in range(self.d_e):
            d_e_search_bounds.append([-np.sqrt(self.d_e), np.sqrt(self.d_e)])
        return np.asarray(d_e_search_bounds)


class rembo_swarm(object):
    """!
    @brief A collection of rembo optimizers.  Each optimizer minimizes the
    target function on a random embedded flat surface
    """
    pass
