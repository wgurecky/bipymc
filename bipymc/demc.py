from bipymc.samplers import *
from mpi4py import MPI


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
        print(rank_chain_ids)
        for i, c_id in enumerate(self.rank_chain_ids):
            self.am_chains.append(McmcChain(theta_0, varepsilon * kwargs.get("inflate", 1e1),
                                  global_id=c_id, mpi_comm=self.comm, mpi_rank=self.comm.rank))

    def _mcmc_run(self, n, theta_0, varepsilon=1e-6, ln_kwargs={}, **kwargs):
        self.local_n_accepted = 0
        self.local_n_rejected = 1
        self._init_ln_like_fn(ln_kwargs)
        # params for DE-MC algo
        dim = len(theta_0)
        gamma = kwargs.get("gamma", 2.38 / np.sqrt(2. * dim))
        delayed_accept = kwargs.get("delayed_accept", True)

        # Init parallel chains
        self._init_chains(theta_0, varepsilon, **kwargs)

        # Parallel DE-MC algo
        j = 0
        while j < (n - self.n_chains):
            banked_prop_array = []
            for i, current_chain in enumerate(self.am_chains):
                # current global chain id
                c_id = current_chain.global_id

                # gather the latest state from all chains
                local_chain_state = []
                for i, local_chain in enumerate(self.am_chains):
                    local_chain_state.append(local_chain.current_pos)
                local_chain_state = np.array(local_chain_state)

                # broadcast current chain state to everyone
                global_chain_state = np.zeros((self.n_chains, len(theta_0)))
                self.comm.Allgather([local_chain_state, MPI.DOUBLE],  # send
                                    [global_chain_state, MPI.DOUBLE]) # recv

                # generate a proposal vector
                # randomly select chain pair from chain pool
                valid_pool_ids = np.delete(np.array(range(self.n_chains)), c_id)
                mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=2)

                mut_a_chain_state = global_chain_state[mut_chain_ids[0]]
                mut_b_chain_state = global_chain_state[mut_chain_ids[1]]

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

                # imediately update chain[i] or bank updates untill all chains complete proposals
                if not delayed_accept:
                    current_chain.append_sample(new_state)
                else:
                    banked_prop_array.append(new_state)
                j += 1
            if delayed_accept:
                for i, current_chain in enumerate(self.am_chains):
                    current_chain.append_sample(banked_prop_array[i])

        # Global accept ratio
        recbuf_n_accepted = np.zeros(self.comm.Get_size(), dtype=int)
        recbuf_n_rejected = np.zeros(self.comm.Get_size(), dtype=int)
        self.comm.Allgather([np.asarray(self.local_n_accepted, dtype=int), MPI.INT],
                            [recbuf_n_accepted, MPI.INT])
        self.comm.Allgather([np.asarray(self.local_n_rejected, dtype=int), MPI.INT],
                            [recbuf_n_rejected, MPI.INT])
        self.n_accepted = np.sum(recbuf_n_accepted)
        self.n_rejected = np.sum(recbuf_n_rejected)


    @property
    def super_chain(self, collection_rank=0):
        """!
        @brief Gather all chains to master rank and append
        their chain states to a global chain state.
        Return global chain state on master rank.
        @param collection_rank int. rank to collect global results
            Defaults to 0 or root rank.
        """
        return self._super_chain(collection_rank)

    def _super_chain(self, collection_rank=0):
        if self.comm.rank == collection_rank:
            super_ch = np.zeros((self.n_chains * self.am_chains[0].chain.shape[0],
                                 self.am_chains[0].chain.shape[1]))
            print("Supper Chain Shape:", super_ch.shape)
            for i, chain in enumerate(self.iter_all_chains(collection_rank)):
                print(chain.chain.shape)
                # super_ch[i::self.n_chains, :] = chain.chain
                super_ch[i::len(self.am_chains), :] = chain.chain
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

    def iter_all_chains(self, collection_rank=0):
        """!
        @brief Global chain generator
        """
        if self.comm.rank == collection_rank:
            for c_id in range(self.n_chains):
                yield self.get_chain(c_id, collection_rank)
        else:
            # everyone else just gets a bunch of None
            for c_id in range(self.n_chains):
                yield None

    def get_chain(self, c_id, collection_rank=0):
        """!
        @brief  Send the desired chain to the desired collection_rank.
        """
        r = self.get_chain_rank(c_id)
        if r == self.comm.rank:
            for chain in self.am_chains:
                if c_id == chain.global_id:
                    return chain
        else:
            if self.comm.rank == collection_rank:
                # wait to recive chain from other rank
                status = MPI.Status()
                recbuf = None
                self.comm.recv(recbuf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                return recbuf
            else:
                assert 0 <= c_id < self.n_chains
                matched_chain = None
                if self.get_chain_rank(c_id) == self.comm.rank:
                    for chain in self.am_chains:
                        if c_id == chain.global_id:
                            matched_chain = chain
                            # send chain to master
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
