from __future__ import print_function, division
from six import iteritems
import numpy as np
from bipymc.samplers import *
from mpi4py import MPI
import sys


class DeMcMpi(DeMc):
    """!
    @brief Parallel impl of DE-MC algo using mpi4py.
    """
    def __init__(self, ln_like_fn, theta_0=None, varepsilon=1e-6, n_chains=8,
                 mpi_comm=MPI.COMM_WORLD, ln_kwargs={}, **kwargs):
        self.comm = mpi_comm
        self.local_n_accepted = 0
        self.local_n_rejected = 1
        if theta_0 is not None:
            self.dim = len(np.asarray(theta_0))
        else:
            self.dim = kwargs.get("dim", 1)
        self.h5_file = kwargs.get("h5_file", "sampler_checkpoint.h5")
        self.warm_start = kwargs.get("warm_start", False)
        self.checkpoint = kwargs.get("checkpoint", 0)
        super(DeMcMpi, self).__init__(ln_like_fn, n_chains, ln_kwargs=ln_kwargs)
        if not self.warm_start:
            self.init_chains(theta_0, varepsilon, **kwargs)
        else:
            self.init_warmstart_chain(self.h5_file)


    def init_chains(self, theta_0, varepsilon=1e-6, **kwargs):
        """!
        @brief Initilize chains from new theta_0 guess
        """
        # distribute chains evenly amoungst the proc ranks
        rank_chain_ids = np.array_split(np.array(range(self.n_chains)), self.comm.size)[self.comm.rank]
        self.rank_chain_ids = rank_chain_ids
        self.am_chains = []
        for i, c_id in enumerate(self.rank_chain_ids):
            self.am_chains.append(McmcChain(theta_0, varepsilon * kwargs.get("inflate", 1e1),
                                  global_id=int(c_id), mpi_comm=self.comm, mpi_rank=self.comm.rank))

    def init_warmstart_chain(self, h5_file):
        """!
        @brief Read chain pool from file
        """
        self.init_chains(np.zeros(self.dim))
        self.load_state(h5_file)

    def _get_local_chain_state(self):
        local_chain_state = []
        for i, local_chain in enumerate(self.am_chains):
            local_chain_state.append(local_chain.current_pos)
        return np.array(local_chain_state)


    def run_mcmc(self, n, **kwargs):
        self._mcmc_run(n, **kwargs)

    def _mcmc_run(self, n, **kwargs):
        # ensure chains are initilized
        if not self.am_chains:
            raise RuntimeError("ERROR: chains not initilized")
        self.local_n_accepted = 0
        self.local_n_rejected = 1
        theta_0 = self.am_chains[0].current_pos
        dim = len(theta_0)
        # demc proposal settings
        gamma_base = kwargs.get("gamma", 2.38 / np.sqrt(2. * dim))
        flip_prob = np.clip(kwargs.get("flip", 0.1), 0.0, 1.0)
        shuffle = kwargs.get("shuffle", True)
        epsilon = kwargs.get("epsilon", 1e-15)

        def update_chain_pool(k, c_id, current_chain, prop_chain_pool, prop_chain_pool_ids):
            """!
            @brief Update the current chain with proposal from prop_chain_pool
            @param k  int current mcmc iteration
            @param c_id int current chain global id
            @param current_chain  bipymc.samplers.McmcChain instance
            @param prop_chain_pool  np_ndarray  proposal states
            """
            valid_pool_ids = np.array(range(len(prop_chain_pool)))
            if c_id in prop_chain_pool_ids:
                valid_idxs = (c_id != prop_chain_pool_ids)
                valid_pool_ids = valid_pool_ids[valid_idxs]

            # DE mutation step
            mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=2)
            mut_a_chain_state = prop_chain_pool[mut_chain_ids[0]]
            mut_b_chain_state = prop_chain_pool[mut_chain_ids[1]]

            # Every 10th step has chance to take large exploration step
            if k % 10 == 0:
                gamma = np.random.choice([gamma_base, 1.0], p=[0.1, 0.9])
            else:
                gamma = gamma_base

            # Generate proposal vector
            prop_vector = gamma * (mut_a_chain_state - mut_b_chain_state)
            prop_vector += current_chain.current_pos
            prop_vector += McmcChain.var_ball(epsilon, dim)

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

        # Parallel DE-MC algo with emcee modification
        j, k_gen = 0, 0
        while j < int((n - self.n_chains) / self.comm.size):
            # chance to flib a/b chain groups
            flip_bool = np.random.choice([True, False], p=[flip_prob, 1 - flip_prob])

            # random chain shuffle order
            shuffle_idx = np.array(range(self.n_chains))
            if shuffle:
                np.random.shuffle(shuffle_idx)

            # gather the latest state from all chains
            local_chain_state = self._get_local_chain_state()

            # broadcast current chain state to everyone
            global_chain_state = np.zeros((self.n_chains, dim))
            self.comm.Allgather([local_chain_state, MPI.DOUBLE],  # send
                                [global_chain_state, MPI.DOUBLE]) # recv
            global_chain_a, global_chain_b = np.array_split(global_chain_state[shuffle_idx], 2)
            global_chain_a_ids, global_chain_b_ids = \
                    np.array_split(shuffle_idx, 2)
            if flip_bool:
                global_chain_a, global_chain_b = global_chain_b, global_chain_a
                global_chain_a_ids, global_chain_b_ids = global_chain_b_ids, global_chain_a_ids

            # update chain a
            for i, current_chain in enumerate(self.am_chains):
                # current global chain id
                c_id = current_chain.global_id
                if c_id not in global_chain_a_ids:
                    continue
                j += 1
                update_chain_pool(j, c_id, current_chain, global_chain_b, global_chain_b_ids)

            # gather the latest state from all chains
            local_chain_state = self._get_local_chain_state()

            # broadcast current chain state to everyone
            global_chain_state = np.zeros((self.n_chains, dim))
            self.comm.Allgather([local_chain_state, MPI.DOUBLE],  # send
                                [global_chain_state, MPI.DOUBLE]) # recv
            global_chain_a, global_chain_b = np.array_split(global_chain_state[shuffle_idx], 2)
            global_chain_a_ids, global_chain_b_ids = \
                    np.array_split(shuffle_idx, 2)
            if flip_bool:
                global_chain_a, global_chain_b = global_chain_b, global_chain_a
                global_chain_a_ids, global_chain_b_ids = global_chain_b_ids, global_chain_a_ids

            # update chain b
            for i, current_chain in enumerate(self.am_chains):
                # current global chain id
                c_id = current_chain.global_id
                if c_id not in global_chain_b_ids:
                    continue
                j += 1
                update_chain_pool(j, c_id, current_chain, global_chain_a, global_chain_a_ids)
            # update number of generations
            k_gen += 1
            self.comm.Barrier()

            # checkpoint the chains
            if self.checkpoint > 0:
                if k_gen % self.checkpoint == 0:
                    self.save_state(self.h5_file)

        # Global accept ratio
        recbuf_n_accepted = np.zeros(self.comm.Get_size(), dtype=int)
        recbuf_n_rejected = np.zeros(self.comm.Get_size(), dtype=int)
        self.comm.Allgather([np.asarray(self.local_n_accepted, dtype=int), MPI.INT],
                            [recbuf_n_accepted, MPI.INT])
        self.comm.Allgather([np.asarray(self.local_n_rejected, dtype=int), MPI.INT],
                            [recbuf_n_rejected, MPI.INT])
        self.n_accepted = np.sum(recbuf_n_accepted)
        self.n_rejected = np.sum(recbuf_n_rejected)
        self.comm.Barrier()

    def save_state(self, h5_file=""):
        """!
        @brief Write chains to H5file.
        Loop over all local chains and write chain state to h5
        Collect all chains to the root process.  This ensures that only the
        root process writes to the HDF5 file so this method works even if hdf5
        was not configured with parallel write enabled.
        """
        if not h5_file:
            h5_file = self.h5_file
        if self.comm.rank == 0:
            h5f = h5py.File(h5_file, "w")
        for chain in self.gather_all_chains(0):
            if self.comm.rank == 0:
                chain.write_chain_h5(h5f)
        if self.comm.rank == 0:
            h5f.close()
        self.comm.Barrier()

    def load_state(self, h5_file=""):
        """!
        @brief Loads chains from H5file.
        """
        if not h5_file:
            h5_file = self.h5_file
        # collective read operation
        with h5py.File(h5_file, 'r') as h5f:
            for chain in self.am_chains:
                chain.read_chain_h5(h5f)
        # all read number of generations
        k_gen = len(self.am_chains[0].chain)
        # ensure all chains have equal length
        for chain in self.am_chains:
            if len(chain.chain) != k_gen:
                raise RuntimeError
        self.comm.Barrier()

    def param_est(self, n_burn, collection_rank=0):
        """!
        @brief Collect all chains on root and
        estimate global chain stats on root rank.
        """
        self.comm.Barrier()
        chain_slice = self.super_chain_mpi(collection_rank)
        if self.comm.rank == collection_rank:
            chain_slice = chain_slice[n_burn:, :]
            mean_theta = np.mean(chain_slice, axis=0)
            std_theta = np.std(chain_slice, axis=0)
            return mean_theta, std_theta, chain_slice
        else:
            return None, None, None

    def super_chain_mpi(self, collection_rank=0):
        """!
        @brief Gather all chains to master rank and append
        their chain states to a global chain state.
        Return global chain state on master rank.
        @param collection_rank int. rank to collect global results
            Defaults to 0 or root rank.
        """
        return self._super_chain(collection_rank)

    def _super_chain(self, collection_rank=0):
        all_chains = self.gather_all_chains(collection_rank)
        if self.comm.rank == collection_rank:
            super_ch = np.zeros((self.n_chains * self.am_chains[0].chain.shape[0],
                                 self.am_chains[0].chain.shape[1]))
            for i, chain in enumerate(all_chains):
                # super_ch[i::len(self.am_chains), :] = chain.chain
                super_ch[i::self.n_chains, :] = chain.chain
            return super_ch
        else:
            return None

    def gather_all_chains(self, collection_rank=0):
        """!
        @brief gather all chains to one rank. By default collect on master
        @param collection_rank int. rank to collect global results
            Defaults to 0 or root rank.
        """
        return list(self.iter_all_chains(collection_rank))

    def iter_local_chains(self):
        """!
        @brief Local chain generator
        """
        for chain in self.am_chains:
            yield chain

    def iter_all_chains(self, collection_rank=0, verbose=0):
        """!
        @brief Global chain generator
        """
        if verbose: print("Iter all chains on rank: ", self.comm.rank)
        sys.stdout.flush()
        for c_id in range(self.n_chains):
            yield self.get_chain(c_id, collection_rank)

    def get_chain(self, c_id, collection_rank=0, verbose=0):
        """!
        @brief  Send the desired chain to the desired collection_rank.
        """
        r = self.get_chain_rank(c_id)
        if r == collection_rank:
            for chain in self.am_chains:
                if c_id == chain.global_id:
                    return chain
        else:
            if self.comm.rank == collection_rank:
                # wait to recive chain from other rank
                status = MPI.Status()
                recbuf = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if verbose: print("Rank: ", self.comm.rank, "Got Chain:", recbuf)
                sys.stdout.flush()
                return recbuf
            else:
                assert 0 <= c_id < self.n_chains
                matched_chain = None
                if r == self.comm.rank:
                    for chain in self.am_chains:
                        if c_id == chain.global_id:
                            matched_chain = chain
                            # send chain to master
                            if verbose:
                                print("Rank: ", self.comm.rank, "Sent a Chain:", matched_chain)
                            sys.stdout.flush()
                            self.comm.send(matched_chain, dest=collection_rank, tag=self.comm.rank)
                return None

    def get_chain_rank(self, c_id):
        """!
        @brief Find rank which houses chain with global id == c_id
        @param c_id int.  Global chain id
        """
        assert 0 <= c_id < self.n_chains
        # get the MPI rank which hold chain with id: c_id
        rank_chain_ids = np.array_split(np.array(range(self.n_chains)), self.comm.size)
        for r, chain_ids in enumerate(rank_chain_ids):
            if c_id in chain_ids:
                return r
        raise RuntimeError("ERROR: c_id not in global chain ids")
