from __future__ import print_function, division
from six import iteritems
from copy import deepcopy
import numpy as np
from bipymc.demc import DeMcMpi
from bipymc.chain import McmcChain
from mpi4py import MPI
import sys


class DreamMpi(DeMcMpi):
    """!
    @brief Parallel impl of DREAM algo using mpi4py.  Extends the DeMcMpi
    implentation with an improved proposal generation algo.
    """
    def __init__(self, ln_like_fn, theta_0=None, varepsilon=1e-6, n_chains=8,
                 mpi_comm=MPI.COMM_WORLD, ln_kwargs={}, **kwargs):
        self.del_pairs = kwargs.get("del_pairs", 40)
        super(DreamMpi, self).__init__(ln_like_fn, theta_0=theta_0, varepsilon=varepsilon, n_chains=n_chains,
                 mpi_comm=mpi_comm, ln_kwargs=ln_kwargs, **kwargs)

    def _update_chain_pool(self, k, c_id, current_chain, prop_chain_pool, prop_chain_pool_ids, **kwargs):
        """!
        @brief Update the current chain with proposal from prop_chain_pool using DREAM
        @param k  int current mcmc iteration
        @param c_id int current chain global id
        @param current_chain  bipymc.samplers.McmcChain instance
        @param prop_chain_pool  np_ndarray  proposal states
        """
        cr_prob = kwargs.get("cr_prob", 0.8)
        epsilon = kwargs.get("epsilon", 1e-15)
        d_prime = deepcopy(self.dim)
        # ensure the current chain is not in the proposal chain pool
        valid_pool_ids = np.array(range(len(prop_chain_pool)))
        if c_id in prop_chain_pool_ids:
            valid_idxs = (c_id != prop_chain_pool_ids)
            valid_pool_ids = valid_pool_ids[valid_idxs]
        del_pairs = np.min((self.del_pairs, int(np.floor(len(valid_pool_ids) / 2))))
        gamma_base = 2.38 / np.sqrt(2. * del_pairs * d_prime)

        # DREAM mutation step
        mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=(2, del_pairs))
        mut_a_chain_state_vec = prop_chain_pool[mut_chain_ids[0, :]]
        mut_b_chain_state_vec = prop_chain_pool[mut_chain_ids[1, :]]

        mut_a_chain_state = np.sum(mut_a_chain_state_vec, axis=0)
        mut_b_chain_state = np.sum(mut_b_chain_state_vec, axis=0)

        # Every 5th step has chance to take large exploration step
        if k % 5 == 0:
            gamma = np.random.choice([gamma_base, 1.0], p=[0.1, 0.9])
        else:
            gamma = gamma_base

        # Generate proposal vector
        eps_u = McmcChain.var_box(epsilon, self.dim)
        eps_n = McmcChain.var_ball(epsilon ** 2.0, self.dim)
        prop_vector = (np.ones(self.dim) + eps_u) * gamma * (mut_a_chain_state - mut_b_chain_state)
        prop_vector += eps_n

        # choose subset of dimensions to update
        # flip unfair coin for each dim
        cr_mask = np.random.choice([True, False], p=[cr_prob, 1.0 - cr_prob], size=self.dim)
        d_prime = np.count_nonzero(cr_mask)
        prop_vector[cr_mask] += current_chain.current_pos[cr_mask]
        prop_vector[~cr_mask] = current_chain.current_pos[~cr_mask]

        # TODO: update crossover probablity

        # Metropolis ratio
        alpha = self._mut_prop_ratio(self._frozen_ln_like_fn,
                                     current_chain.current_pos,
                                     prop_vector)
        if self.metropolis_accept(alpha):
            new_state = prop_vector
            self.local_n_accepted += 1
        else:
            new_state = current_chain.current_pos
            self.local_n_rejected += 1

        # imediately update chain[i]
        current_chain.append_sample(new_state)
