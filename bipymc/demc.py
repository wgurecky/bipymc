from bipymc.samplers import *
from mpi4py import MPI
import sys


class DeMcMpi(DeMc):
    """!
    @brief Parallel impl of DE-MC algo using mpi4py
    This method works well when the liklihood function is expensive to update.
    For very cheep liklihood functions the communication overhead might
    reduce overall performance.
    """
    def __init__(self, ln_like_fn, n_chains=8, mpi_comm=MPI.COMM_WORLD):
        self.comm = mpi_comm
        # self.comm_rank = mpi_comm.Get_rank()
        # self.comm_size = mpi_comm.Get_size()
        self.local_n_accepted = 0
        self.local_n_rejected = 1
        super(DeMcMpi, self).__init__(ln_like_fn, n_chains)

    def _init_chains(self, theta_0, varepsilon=1e-6, **kwargs):
        # initilize chains
        # distribute chains evenly amoungst the proc ranks
        rank_chain_ids = np.array_split(np.array(range(self.n_chains)), self.comm.size)[self.comm.rank]
        self.rank_chain_ids = rank_chain_ids
        self.am_chains = []
        for i, c_id in enumerate(self.rank_chain_ids):
            self.am_chains.append(McmcChain(theta_0, varepsilon * kwargs.get("inflate", 1e1),
                                  global_id=c_id, mpi_comm=self.comm, mpi_rank=self.comm.rank))

    def _get_local_chain_state(self):
        local_chain_state = []
        for i, local_chain in enumerate(self.am_chains):
            local_chain_state.append(local_chain.current_pos)
        return np.array(local_chain_state)

    def _mcmc_run(self, n, theta_0, varepsilon=1e-6, ln_kwargs={}, **kwargs):
        self.local_n_accepted = 0
        self.local_n_rejected = 1
        self._init_ln_like_fn(ln_kwargs)
        # params for DE-MC algo
        dim = len(theta_0)
        gamma = kwargs.get("gamma", 2.38 / np.sqrt(2. * dim))
        delayed_accept = kwargs.get("delayed_accept", True)


        def update_chain_pool(current_chain, chain_pool):
            # valid_pool_ids = np.delete(np.array(range(self.n_chains)), c_id)
            valid_pool_ids =np.array(range(len(chain_pool)))
            mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=2)
            mut_a_chain_state = chain_pool[mut_chain_ids[0]]
            mut_b_chain_state = chain_pool[mut_chain_ids[1]]

            # generate proposal vector
            prop_vector = gamma * (mut_a_chain_state - mut_b_chain_state)
            prop_vector += current_chain.current_pos
            prop_vector += McmcChain.var_ball(varepsilon * 1e-3, dim)

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

        # Init parallel chains
        self._init_chains(theta_0, varepsilon, **kwargs)

        # Parallel DE-MC algo with emcee modification
        j = 0
        while j < int((n - self.n_chains) / self.comm.size):
            flip_bool = np.random.choice([True, False], p=[0.5, 0.5])

            # gather the latest state from all chains
            local_chain_state = self._get_local_chain_state()

            # broadcast current chain state to everyone
            global_chain_state = np.zeros((self.n_chains, len(theta_0)))
            self.comm.Allgather([local_chain_state, MPI.DOUBLE],  # send
                                [global_chain_state, MPI.DOUBLE]) # recv
            global_chain_a, global_chain_b = np.array_split(global_chain_state, 2)
            global_chain_a_ids, global_chain_b_ids = \
                    np.array_split(np.array(range(self.n_chains)), 2)
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
                update_chain_pool(current_chain, global_chain_b)

            # gather the latest state from all chains
            local_chain_state = self._get_local_chain_state()

            # broadcast current chain state to everyone
            global_chain_state = np.zeros((self.n_chains, len(theta_0)))
            self.comm.Allgather([local_chain_state, MPI.DOUBLE],  # send
                                [global_chain_state, MPI.DOUBLE]) # recv
            global_chain_a, global_chain_b = np.array_split(global_chain_state, 2)
            global_chain_a_ids, global_chain_b_ids = \
                    np.array_split(np.array(range(self.n_chains)), 2)
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
                update_chain_pool(current_chain, global_chain_a)

            self.comm.Barrier()

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
