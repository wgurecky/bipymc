from __future__ import print_function, division
from six import iteritems
import h5py
import numpy as np
import scipy.stats as stats


class McmcChain(object):
    """!
    @brief A simple mcmc chain with some helper functions.
    """
    def __init__(self, theta_0, varepsilon=1e-6, global_id=0, mpi_comm=None, mpi_rank=None):
        """!
        @brief MCMC chain init
        @param theta_0 array like.  initial guess for parameters.
        @param varepsilon float  var of gaussian noise used to initilize chain state:
            \f[ \thata_0 + \mathcal E;   \mathcal E \sim \mathcal N(0, \varepsilon) \f]
        @param global_id int  chain id
        """
        assert isinstance(global_id, int)
        assert varepsilon >= 0.0
        assert global_id >= 0
        self.global_id = global_id
        theta_0_flat = np.asarray(theta_0).flatten()
        self._dim = len(theta_0_flat)
        theta_0_flat = np.asarray(theta_0_flat) + self.var_ball(varepsilon, self._dim)
        # init the 2d array of shape (iteration, <theta_vec>)
        self._chain = np.array([theta_0_flat])

    @staticmethod
    def var_ball(varepsilon, dim):
        """!
        @brief Draw single sample from tight gaussian ball
        """
        var_epsilon = 0.
        if varepsilon > 0:
            var_epsilon = np.random.multivariate_normal(np.zeros(dim),
                                                        np.eye(dim) * varepsilon,
                                                        size=1)[0]
        return var_epsilon

    def set_t_kernel(self, t_kernel):
        """!
        @brief Set valid transition kernel
        """
        assert t_kernel.shape[1] == self._chain.shape[1]
        assert t_kernel.shape[1] == t_kernel.shape[0]  # must be square
        self.t_kernel = t_kernel

    def t_kernel_eig(self):
        """!
        @brief Get eigenvals of transition kernel
        """
        return np.linalg.eig(self.t_kernel)

    def apply_t_kernel(self, apply_new_state=True):
        new_state = np.dot(self.t_kernel, self._chain[:-1])
        if apply_new_state:
            self.append_sample(new_state)
        return new_state

    def append_sample(self, theta_new):
        theta_new = np.asarray(theta_new)
        assert theta_new.shape[0] == self.chain.shape[1]
        self._chain = np.vstack((self._chain, theta_new))

    def pop_sample(self):
        self._chain = self._chain[:-1, :]

    def write_chain_h5(self, h5_file):
        """!
        @brief Write chain state to h5 file
        @param h5_file either str or h5py.File instance
        """
        if isinstance(h5_file, str):
            with h5py.File(h5_file, 'w') as h5f:
                try:
                    del h5f["/chains/chain_id_" + str(self.global_id)]
                except:
                    pass
                h5f.create_dataset("/chains/chain_id_" + str(self.global_id), data=self.chain,
                                   compression="gzip")
        elif isinstance(h5_file, h5py.File):
            try:
                del h5_file["/chains/chain_id_" + str(self.global_id)]
            except:
                pass
            h5_file.create_dataset("/chains/chain_id_" + str(self.global_id), data=self.chain,
                                   compression="gzip")
        else:
            raise RuntimeError

    def read_chain_h5(self, h5_file, c_id=None):
        """!
        @brief Load chain state from h5 file
        @param h5_file either str or h5py.File instance
        """
        if isinstance(h5_file, str):
            with h5py.File(h5_file, 'r') as h5f:
                self.load_chain_state(h5f["/chains/chain_id_" + str(self.global_id)][:])
        elif isinstance(h5_file, h5py.File):
            self.load_chain_state(h5_file["/chains/chain_id_" + str(self.global_id)][:])
        else:
            raise RuntimeError

    def load_chain_state(self, chain_state):
        self.chain = chain_state

    def auto_corr(self, lag):
        """!
        @brief Compute auto correlation in the chain.
        """
        pass

    @property
    def chain(self):
        return self._chain

    @chain.setter
    def chain(self, input_chain):
        assert input_chain.shape[1] == self._dim
        self._chain = input_chain

    @property
    def current_pos(self):
        return self.chain[-1, :]

    @property
    def chain_len(self):
        return self.chain.shape[0]

    @property
    def dim(self):
        return self._dim


