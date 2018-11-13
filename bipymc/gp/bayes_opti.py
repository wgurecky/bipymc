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
import sys
from copy import deepcopy
import numpy as np
from gpr import *


class bo_optimizer(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self, f, dim, p_bounds, x0=None, y0=None, n_init=2,
                 fn_args=[], fn_kwargs={}, comm=MPI.COMM_WORLD, **kwargs):
        # minimize or maximize fn flag
        self.minimize = kwargs.get("min", True)
        self.comm = comm
        # setup obj function and parameter bounds
        self.dim = dim
        self._p_bounds = None
        self.param_bounds = p_bounds
        self.obj_f = lambda x: f(x, *fn_args, **fn_kwargs)
        self._init_gp(x0, y0, dim, n_init, **kwargs)

    def _init_gp(self, x0, y0, dim, n_init, **kwargs):
        # setup gp model
        self.gp_model = gp_regressor(ndim=dim)
        # eval obj fn n_init times to seed the method
        if x0 is not None:
            self.x_known = x0
            self.y_known = y0
        else:
            self.x_known, self.y_known = self.sample_uniform(n_init)
        # fit the gp model to initial seed points
        self.y_sigma = kwargs.get("y_sigma", 1e-8)
        self.gp_model.fit(self.x_known, self.y_known.flatten(), self.y_sigma)

    @property
    def param_bounds(self):
        return self._p_bounds

    @param_bounds.setter
    def param_bounds(self, p_bounds):
        """!
        @brief set parameter bounds
        Note: upper and lower vals req for each param
        """
        assert isinstance(p_bounds, (list, tuple))
        assert len(p_bounds) == self.dim
        assert isinstance(p_bounds[0], (list, tuple))
        assert len(p_bounds[0]) == 2
        self._p_bounds = list(list(p_b) for p_b in p_bounds)

    def optimize(self, n_iter=10, n_samples=40, max_depth=2):
        for i in range(n_iter):
            self.comm.Barrier()
            # everyone get a proposal sample
            converged = False
            while not converged:
                try:
                    x_proposal, y_proposal = self.sample_thompson(max_depth=max_depth, n=n_samples)
                    converged = True
                except:
                    print("Resampling")
                    sys.stdout.flush()

            # update known sample locations
            self.x_known = np.vstack((self.x_known, x_proposal))
            self.y_known = np.concatenate((self.y_known, y_proposal))

            # fit GP to current data
            self.gp_model.fit(self.x_known, self.y_known.flatten(), self.y_sigma)

            # print mins
            self.comm.Barrier()
            if self.comm.rank == 0:
                best_idx = np.argmin(self.y_known)
                print("Iteration: %d, Best Obj Fn val=%0.5e at %s" % \
                        (i, self.y_known[best_idx], str(self.x_known[best_idx])))
                sys.stdout.flush()

    def sample_uniform(self, n_init=2, random=True):
        sample_grid = []
        p_bounds = deepcopy(self.param_bounds)
        for bounds in p_bounds:
            rand_x = np.random.uniform(bounds[0], bounds[1], n_init)
            sample_grid.append(rand_x)
        x_grid = np.array(sample_grid).T
        response = self.obj_f(x_grid).flatten()
        return x_grid, response

    def sample_thompson(self, max_depth=2, n=40, mode='min', diag_scale=1e-6):
        assert mode in ('min', 'max', 'explore')
        p_bounds = deepcopy(self.param_bounds)
        for i in range(max_depth):
            sample_grid, grid_div_size = [], []
            for bounds in p_bounds:
                start, end = bounds[0], bounds[1]
                search_grid = np.random.uniform(bounds[0], bounds[1], n)
                div_size = np.abs(np.std(search_grid))
                grid_div_size.append(div_size)
                sample_grid.append(search_grid)
            x_mesh = np.meshgrid(*sample_grid)
            x_mesh_flat = []
            for x_ in x_mesh:
                x_mesh_flat.append(x_.flatten())
            sample_x = np.array(x_mesh_flat).T
            if i == 0:
                sample_y = self.gp_model.sample_y(sample_x, diag_scale=diag_scale)
            else:
                if mode == 'min':
                    sample_y = self.gp_model.predict(sample_x).flatten() - \
                        self.gp_model.predict_sd(sample_x).flatten()
                if mode == 'max':
                    sample_y = self.gp_model.predict(sample_x).flatten() + \
                        self.gp_model.predict_sd(sample_x).flatten()
            if mode == 'min':
                # pick location with lowest response
                idx_best = np.argmin(sample_y)
            elif mode == 'max':
                # pick location with highest response
                idx_best = np.argmax(sample_y)
            else:
                # pick location with most varience in response surface
                sample_yn = self.gp_model.sample_y(sample_x, n_draws=10, diag_scale=diag_scale)
                idx_best = np.argmax(np.var(sample_yn, axis=1))
            best_x = sample_x[idx_best]
            best_y = sample_y[idx_best]
            # collapse p_bounds about best est
            for j in range(len(p_bounds)):
                p_bounds[j][0] = np.clip(best_x[j] - 0.49 * grid_div_size[j], \
                        self.param_bounds[j][0], self.param_bounds[j][1])
                p_bounds[j][1] = np.clip(best_x[j] + 0.49 * grid_div_size[j], \
                        self.param_bounds[j][0], self.param_bounds[j][1])
            n /= 2
        # broadcast
        y_new = self.obj_f(np.array([best_x]))
        all_best_x = np.zeros((self.comm.size, self.dim)).flatten()
        all_best_y = np.zeros((self.comm.size))
        self.comm.Allgather(best_x.flatten(), all_best_x)
        all_best_x = all_best_x.reshape((self.comm.size, self.dim))
        self.comm.Allgather(np.array(y_new), all_best_y)
        return all_best_x, all_best_y.flatten()


def one_dim_ex():
    def obj_fn_sin(Xtest):
        return np.sin(Xtest)

    # bounds on params (x, y)
    my_bounds = ((-4, 4),)
    my_bo = bo_optimizer(obj_fn_sin, dim=1, p_bounds=my_bounds, n_init=2)

    # run optimizer
    my_bo.optimize(20)

    # plot the response surface
    plt.figure()
    x_test = np.linspace(my_bounds[0][0], my_bounds[0][1], 100)
    plt.plot(x_test, my_bo.gp_model.predict(x_test), label="mean")
    plt.scatter(my_bo.x_known, my_bo.y_known, label="samples", c='k', s=10, marker='x')
    plt.grid(axis='both', ls='--', alpha=0.5)
    plt.legend()
    plt.savefig("bo_sin_test.png")
    plt.close()

def two_dim_ex():
    def obj_fn_2d(Xtest):
        X, Y = Xtest[:, 0], Xtest[:, 1]
        return X ** 2.0 + Y ** 2.0

    # plot original fn
    x1 = np.linspace(-4, 4, 20)
    y1 = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(x1, y1)
    Xtest = np.array((X.flatten(), Y.flatten())).T
    Z = obj_fn_2d(Xtest)
    Z = Z.reshape(X.shape)

    # plot contour
    contour_plot = plt.figure()
    x_grid, y_grid, z_grid = X, Y, Z
    plt.subplot(1, 1, 1)
    nlevels = 20
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, nlevels, colors='k', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.scatter(X, Y, c='k', s=3, alpha=0.6)
    contour_plot.savefig("bo_2d_orig.png")

    # bounds on params (x, y)
    my_bounds = ((-4, 4), (-4, 4))
    my_bo = bo_optimizer(obj_fn_2d, dim=2, p_bounds=my_bounds, n_init=1, y_sigma=1e-1)

    # run optimizer
    my_bo.optimize(20)
    # best estimate
    plt.scatter(my_bo.x_known[:, 0], my_bo.x_known[:, 1], c='r', s=4, alpha=0.8)
    contour_plot.savefig("bo_2d_sampled.png")
    plt.close()

    n = 50
    my_gpr_nd = my_bo.gp_model
    Xtest = np.linspace(-5, 5, n)
    xt, yt = np.meshgrid(Xtest, Xtest)
    Xtest = np.array((xt.flatten(), yt.flatten())).T
    ytest_mean = my_gpr_nd.predict(Xtest)
    zt = ytest_mean.reshape(xt.shape)

    # plot contour
    contour_plot = plt.figure()
    x_grid, y_grid, z_grid = xt, yt, zt
    plt.subplot(1, 1, 1)
    nlevels = 20
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, nlevels, colors='k', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.scatter(my_bo.x_known[:, 0], my_bo.x_known[:, 1], c='r', s=4, alpha=0.8)
    contour_plot.savefig("bo_2d_predicted.png")
    print("Total N samples: %d" % len(my_bo.y_known))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    one_dim_ex()
    two_dim_ex()
