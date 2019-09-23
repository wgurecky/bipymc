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
        self.del_pairs = kwargs.get("del_pairs", 3)
        self.n_cr = kwargs.get("n_cr", 3)
        super(DreamMpi, self).__init__(ln_like_fn, theta_0=theta_0, varepsilon=varepsilon, n_chains=n_chains,
                 mpi_comm=mpi_comm, ln_kwargs=ln_kwargs, **kwargs)
        self._init_cr()

    def _update_chain_pool(self, k, c_id, current_chain, prop_chain_pool, prop_chain_pool_ids, **kwargs):
        """!
        @brief Update the current chain with proposal from prop_chain_pool using DREAM
        @param k  int current mcmc iteration
        @param c_id int current chain global id
        @param current_chain  bipymc.samplers.McmcChain instance
        @param prop_chain_pool  np_ndarray  proposal states
        """
        epsilon = kwargs.get("epsilon", 1e-12)
        u_epsilon = kwargs.get("u_epsilon", 1e-2)
        # ensure the current chain is not in the proposal chain pool
        valid_pool_ids = np.array(range(len(prop_chain_pool)))
        if c_id in prop_chain_pool_ids:
            valid_idxs = (c_id != prop_chain_pool_ids)
            valid_pool_ids = valid_pool_ids[valid_idxs]
        # del_pairs = np.min((self.del_pairs, int(np.floor(len(valid_pool_ids) / 2))))

        # choose subset of dimensions to update
        # flip unfair coin for each dim
        cr = np.random.choice(self.CR, p=self.p_cr, size=1)
        z = np.random.uniform(0, 1, size=self.dim)
        cr_mask = (z <= cr)
        # if no dimension was selected for update, choose one at random
        if np.count_nonzero(cr_mask) == 0:
            rand_idx = np.random.choice(range(len(cr_mask)))
            cr_mask[rand_idx] = True
        d_prime = np.count_nonzero(cr_mask)

        # DREAM mutation step
        gamma_base = 2.38 / np.sqrt(2. * self.del_pairs * d_prime)
        mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=(2, self.del_pairs))
        mut_a_chain_state_vec = prop_chain_pool[mut_chain_ids[0, :]]
        mut_b_chain_state_vec = prop_chain_pool[mut_chain_ids[1, :]]

        mut_a_chain_state = np.sum(mut_a_chain_state_vec, axis=0)
        mut_b_chain_state = np.sum(mut_b_chain_state_vec, axis=0)
        assert len(mut_a_chain_state) == len(mut_b_chain_state)
        update_dims = np.zeros(len(mut_a_chain_state))
        update_dims[cr_mask] = 1.0

        # Every 10th step has chance to take large exploration step
        if k % 10 == 0:
            gamma = np.random.choice([gamma_base, 1.0], p=[0.20, 0.80])
        else:
            gamma = gamma_base

        # Generate proposal vector
        eps_u = McmcChain.var_box(u_epsilon, self.dim)
        eps_n = McmcChain.var_ball(epsilon ** 2.0, self.dim)
        prop_vector = (np.ones(self.dim) + eps_u) * gamma * \
                (mut_a_chain_state - mut_b_chain_state) * update_dims
        prop_vector += eps_n

        # update proposal vector only in select dimensions
        prop_vector += current_chain.current_pos

        # update crossover probablity
        if self.in_burnin:
            self._update_cr_ratios(current_chain, prop_vector, cr)

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

    def _init_cr(self):
        """!
        @brief Initilizes the crossover prob vector.
        """
        self.CR = (np.array(range(self.n_cr)) + 1) / self.n_cr
        self.p_cr = np.ones(self.n_cr) / self.n_cr
        self.n_cr_updates = np.zeros(len(self.CR))
        self.p_cr_update = np.zeros(len(self.CR))
        self.delta_m = np.zeros(len(self.CR))

    def _update_cr_ratios(self, current_chain, prop_vector, cr):
        """!
        @brief Adapt crossover probs, p_cr
        """
        n_samples = len(current_chain.chain)
        if n_samples > 200:
            cr_idx = np.where(self.CR == cr)
            self.n_cr_updates[cr_idx] += 1.0

            std_devs = np.std(current_chain.chain, axis=0)
            std_devs[std_devs == 0] = 1e-12
            delta_m_update = np.sum(((current_chain.current_pos - prop_vector) ** 2.0 / std_devs ** 2.0))

            self.delta_m[cr_idx] += delta_m_update

            if np.count_nonzero(self.n_cr_updates) == self.n_cr:
                for m in range(self.n_cr):
                    self.p_cr_update[m] = (self.delta_m[m] / self.n_cr_updates[m])
                self.p_cr = self.p_cr_update

            # normalize
            self.p_cr /= np.sum(self.p_cr)
            print(self.p_cr)

    @property
    def in_burnin(self):
        return True
