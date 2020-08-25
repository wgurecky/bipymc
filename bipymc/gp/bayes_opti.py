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
from __future__ import division
from mpi4py import MPI
from itertools import product
import time
import sys
from copy import deepcopy
import numpy as np
from scipydirect import minimize as ncsu_direct_min
from bipymc.gp.gpr import *


class bo_optimizer(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self, f, dim, s_bounds, x0=None, y0=None, n_init=2,
                 fn_args=[], prior=None, fn_kwargs={}, comm=MPI.COMM_WORLD, **kwargs):
        # minimize or maximize fn flag
        self.gp_fit_kwargs = kwargs.get("gp_fit_kwargs", {})
        assert isinstance(self.gp_fit_kwargs, dict)
        self.minimize = kwargs.get("min", True)
        self.comm = comm
        # setup obj function and parameter bounds
        self.dim = dim
        self.search_bounds = s_bounds
        self.obj_f = lambda x: f(x, *fn_args, **fn_kwargs)
        if isinstance(prior, gp_regressor):
            assert prior.is_fit
            self.y_sigma = kwargs.get("y_sigma", 1e-8)
            self.gp_model = prior
        else:
            self._init_gp(x0, y0, dim, n_init, **kwargs)

    @property
    def x_known(self):
        return self._x_known

    @x_known.setter
    def x_known(self, x_known):
        self._x_known = x_known

    def _init_gp(self, x0, y0, dim, n_init, **kwargs):
        # setup gp model
        self.gp_model = gp_regressor(ndim=dim, domain_bounds=self.search_bounds)
        # eval obj fn n_init times to seed the method
        if x0 is not None:
            self.x_known = x0
            self.y_known = y0
        else:
            self.x_known, self.y_known = self.sample_uniform(n_init)
        # fit the gp model to initial seed points
        self.y_sigma = kwargs.get("y_sigma", 1e-8)
        self.gp_model.fit(self.x_known, self.y_known.flatten(),
                self.y_sigma, **self.gp_fit_kwargs)

    @property
    def search_bounds(self):
        return self._s_bounds

    @search_bounds.setter
    def search_bounds(self, s_bounds):
        """!
        @brief set parameter bounds
        Note: upper and lower vals req for each param
        """
        assert isinstance(s_bounds, (list, tuple))
        assert len(s_bounds) == self.dim
        assert isinstance(s_bounds[0], (list, tuple))
        assert len(s_bounds[0]) == 2
        self._s_bounds = list(list(p_b) for p_b in s_bounds)

    def optimize(self, n_iter=10, n_samples=100, max_depth=2, mode='min', diag_scale=1e-6,
                 method='direct', return_y=False):
        for i in range(n_iter):
            self.comm.Barrier()
            # everyone get a proposal sample
            converged, try_count = False, 0
            while not converged:
                try:
                    if method == 'direct':
                        x_proposal, y_proposal = self.sample_thompson_direct( \
                                n=n_samples, mode=mode, diag_scale=diag_scale)
                    else:
                        x_proposal, y_proposal = self.sample_thompson( \
                                max_depth=max_depth, n=n_samples, mode=mode, diag_scale=diag_scale)
                    converged = True
                except:
                    try_count += 1
                    if try_count > 1:
                        print("WARNING: Thompson sample failed. Re-Sampling.")
                        x_proposal, y_proposal = self.sample_uniform(1)
                        break
            # broadcase samples
            all_best_x = np.zeros((self.comm.size, self.dim)).flatten()
            all_best_y = np.zeros((self.comm.size))
            self.comm.Allgather(x_proposal.flatten(), all_best_x)
            all_best_x = all_best_x.reshape((self.comm.size, self.dim))
            self.comm.Allgather(np.array(y_proposal), all_best_y)
            all_x_proposal, all_y_proposal = all_best_x, all_best_y.flatten()

            # update known sample locations
            self.x_known = np.vstack((self.x_known, all_x_proposal))
            self.y_known = np.concatenate((self.y_known, all_y_proposal))

            # fit GP to current data
            self.gp_model.fit(self.x_known, self.y_known.flatten(), self.y_sigma, **self.gp_fit_kwargs)

            # print mins
            if self.comm.rank == 0:
                if mode == 'min':
                    best_idx = np.argmin(self.y_known)
                else:
                    best_idx = np.argmax(self.y_known)
                print("Iteration: %d, Best Obj Fn val=%0.5e at %s" % \
                        (i, self.y_known[best_idx], str(self.x_known[best_idx])))
                sys.stdout.flush()
        self.comm.Barrier()
        if mode == 'min':
            best_idx = np.argmin(self.y_known)
        else:
            best_idx = np.argmax(self.y_known)
        if return_y:
            return self.x_known[best_idx], self.y_known[best_idx]
        return self.x_known[best_idx]

    def sample_thompson_direct(self, n=400, mode='min', diag_scale=1e-6, **kwargs):
        assert mode in ('min', 'max', 'explore')
        # define surrogate surface as draw from GP model
        if mode == 'min':
            gp_sf = lambda x: self.gp_model.sample_y(np.asarray([x]), n_draws=1,
                    diag_scale=diag_scale, rnd_seed=self.comm.rank).T[0]
        elif mode == 'max':
            gp_sf = lambda x: -1.0 * (kwargs.get('y_shift', 1e8) + \
                    self.gp_model.sample_y(np.asarray([x]), n_draws=1, \
                    diag_scale=diag_scale, rnd_seed=self.comm.rank).T[0])
        else:
            gp_sf = lambda x: -1.0 * np.var(self.gp_model.sample_y(np.asarray([x]), n_draws=10,
                    diag_scale=diag_scale, rnd_seed=self.comm.rank).flatten())
        sb = []
        for bounds in self.search_bounds:
            bounds_scale = np.abs(bounds[1] - bounds[0]) * kwargs.get("bounds_jitter", 0.05)
            shrinkage = np.random.uniform(low=0, high=1.0, size=2) * bounds_scale
            b_low = bounds[0] + shrinkage[0]
            b_high = bounds[1] - shrinkage[1]
            sb.append([b_low, b_high])
        res = ncsu_direct_min(gp_sf, bounds=sb,
                              maxf=n, algmethod=kwargs.get('algmethod', 1))
        best_x = res.x
        # add jitter to prevent possibility of two samples landing in same loc
        x_purt = kwargs.get("x_purt", 1e-8)
        best_x += np.random.uniform(-x_purt, x_purt, 1)
        best_y = self.obj_f(np.array([best_x]))
        return best_x, best_y

    def sample_uniform(self, n_init=2, random=True):
        sample_grid = []
        s_bounds = deepcopy(self.search_bounds)
        for bounds in s_bounds:
            rand_x = np.random.uniform(bounds[0], bounds[1], n_init)
            sample_grid.append(rand_x)
        x_grid = np.array(sample_grid).T
        response = self.obj_f(x_grid).flatten()
        return x_grid, response

    def _chunked_meshgrid(self, *args):
        return product(*args)

    def _gen_sample_grid(self, s_bounds, n):
        sample_grid, grid_div_size = [], []
        for bounds in s_bounds:
            start, end = bounds[0], bounds[1]
            search_grid = np.random.uniform(bounds[0], bounds[1], int(n))
            div_size = np.abs(np.std(search_grid))
            grid_div_size.append(div_size)
            sample_grid.append(search_grid)
        return sample_grid, grid_div_size

    def sample_thompson(self, max_depth=2, n=100, mode='min', diag_scale=1e-6):
        assert mode in ('min', 'max', 'explore')
        s_bounds = deepcopy(self.search_bounds)
        for depth in range(max_depth):
            sample_grid, grid_div_size = self._gen_sample_grid(s_bounds, n)

            # chunked mesh grid, generate 100 points at a time
            x_mesh_chunk, n_c = [], int(n ** self.dim)
            results_x, results_y = [], []
            for c, grid_coord in enumerate(self._chunked_meshgrid(*sample_grid)):
                x_mesh_chunk.append(grid_coord)
                if (c + 1) % 50 == 0 or c == n_c - 1:
                    tmp_x_chunk = np.array(x_mesh_chunk)
                    tmp_results_x, tmp_results_y = self._gen_thompson_samples( \
                            depth, tmp_x_chunk, s_bounds, n, mode, diag_scale, grid_div_size)
                    results_x.append(tmp_results_x)
                    results_y.append(tmp_results_y)
                    x_mesh_chunk = []
            chunked_x, chunked_y = np.array(results_x).reshape(-1, self.dim), np.array(results_y).reshape(-1, 1)
            # get best overall result from all chunks
            idx_best = self._pick_best_idx(chunked_y, mode)
            best_x = chunked_x[idx_best]
            # collapse s_bounds about best est
            for j in range(len(s_bounds)):
                s_bounds[j][0] = np.clip(best_x[j] - 0.4 * grid_div_size[j], \
                        self.search_bounds[j][0], self.search_bounds[j][1])
                s_bounds[j][1] = np.clip(best_x[j] + 0.4 * grid_div_size[j], \
                        self.search_bounds[j][0], self.search_bounds[j][1])
            # reduce sample size
            print("Depth=%d, sample_size = %d, max grid size=%f" % (depth, n, np.max(grid_div_size)))
            n /= 2
            if n < 3:
                break
        # add jitter to prevent possibility of two samples landing in same loc
        best_x += np.random.uniform(-1e-8, 1e-8, 1)
        best_y = self.obj_f(np.array([best_x]))
        return best_x, best_y

    def _pick_best_idx(self, sample_y, mode):
        if mode == 'min':
            # pick location with lowest response
            idx_best = np.argmin(sample_y)
        elif mode == 'max':
            # pick location with highest response
            idx_best = np.argmax(sample_y)
        else:
            # pick location with most varience in response surface
            idx_best = np.argmax(sample_y)
        return idx_best

    def _gen_thompson_samples(self, depth, x_mesh_flat, s_bounds, n, mode, diag_scale, grid_div_size):
            sample_x = x_mesh_flat # np.array(x_mesh_flat).T
            if mode == 'explore':
                # pick location with most varience in response surface
                sample_yn = self.gp_model.sample_y(sample_x, n_draws=10, diag_scale=diag_scale, chunk_size=1e12)
                idx_best = np.argmax(np.var(sample_yn, axis=1))
                best_y = self.gp_model.predict_sd(sample_x[idx_best], chunk_size=1e12)
            else:
                if depth == 0:
                    sample_y = self.gp_model.sample_y(sample_x, diag_scale=diag_scale, chunk_size=1e12)
                else:
                    sample_y = self.gp_model.predict(sample_x)
                if mode == 'min':
                    # pick location with lowest response
                    idx_best = np.argmin(sample_y)
                    best_y = sample_y[idx_best]
                else:
                    # pick location with highest response
                    idx_best = np.argmax(sample_y)
                    best_y = sample_y[idx_best]
            best_x = sample_x[idx_best]
            return best_x, best_y


def one_dim_ex():
    comm = MPI.COMM_WORLD
    def obj_fn_sin(Xtest):
        return np.sin(Xtest)

    # bounds on params (x, y)
    my_bounds = ((-4, 4),)
    my_bo = bo_optimizer(obj_fn_sin, dim=1, s_bounds=my_bounds, n_init=2, comm=comm)

    # run optimizer
    my_bo.optimize(20)

    # plot the response surface
    if comm.rank == 0:
        plt.figure()
        x_test = np.linspace(my_bounds[0][0], my_bounds[0][1], 100)
        plt.plot(x_test, my_bo.gp_model.predict(x_test), label="mean")
        plt.scatter(my_bo.x_known, my_bo.y_known, label="samples", c='k', s=10, marker='x')
        plt.grid(axis='both', ls='--', alpha=0.5)
        plt.legend()
        plt.savefig("bo_sin_test.png")
        plt.close()

def two_dim_ex():
    comm = MPI.COMM_WORLD
    def obj_fn_2d(Xtest):
        X, Y = Xtest[:, 0], Xtest[:, 1]
        # time.sleep(1)
        return X ** 2.0 + Y ** 2.0

    # plot original fn
    x1 = np.linspace(-4, 4, 20)
    y1 = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(x1, y1)
    Xtest = np.array((X.flatten(), Y.flatten())).T
    Z = obj_fn_2d(Xtest)
    Z = Z.reshape(X.shape)

    # plot contour
    if comm.rank == 0:
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
        plt.close()

    # bounds on params (x, y)
    my_bounds = ((-4, 4), (-4, 4))
    my_bo = bo_optimizer(obj_fn_2d, dim=2, s_bounds=my_bounds, n_init=1, y_sigma=1e-1, comm=comm)

    # run optimizer
    my_bo.optimize(20, n_samples=20, max_depth=4)
    # best estimate
    if comm.rank == 0:
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
    if comm.rank == 0:
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
        plt.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    one_dim_ex()
    two_dim_ex()
