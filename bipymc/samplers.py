from __future__ import print_function, division
from six import iteritems
import numpy as np
import scipy.stats as stats
from bipymc.proposals import *
from bipymc.chain import *


class McmcSampler(object):
    """!
    @brief Markov Chain Monte Carlo (MCMC) base class.
    Contains common methods for dealing with the likelyhood function
    and for obtaining parameter estimates from the mcmc chain.
    """
    def __init__(self, log_like_fn, ln_kwargs={}, proposal="Gauss", **proposal_kwargs):
        """!
        @brief setup mcmc sampler
        @param log_like_fn callable.  must return log likelyhood (float)
        @param proposal string.
        @param proposal_kwargs dict.
        """
        self.am_chains = []
        self.log_like_fn = log_like_fn
        if proposal == 'Gauss':
            frozen_ln_like_fn = self._freeze_ln_like_fn(**ln_kwargs)
            self.mcmc_proposal = GaussianProposal(frozen_ln_like_fn, proposal_kwargs)
        else:
            raise RuntimeError("ERROR: proposal type: %s not supported." % str(proposal))
        self.n_accepted = 1
        self.n_rejected = 0

    def _init_chains(self, theta_0, **kwargs):
        raise NotImplementedError

    def _freeze_ln_like_fn(self, **kwargs):
        """!
        @brief Freezes the likelyhood function.
        the log_like_fn should have signature:
            self.log_like_fn(theta, data=np.array([...]), **kwargs)
            and must return the log likelyhood
        """
        self._frozen_ln_like_fn = lambda theta: self.log_like_fn(theta, **kwargs)

    @property
    def frozen_ln_like_fn(self):
        return self._freeze_ln_like_fn

    def param_est(self, n_burn):
        """!
        @brief Computes mean an std of sample chain discarding the first n_burn samples.
        @return  mean (np_1darray), std (np_1darray), chain (np_ndarray)
        """
        chain_slice = self.chain[n_burn:, :]
        mean_theta = np.mean(chain_slice, axis=0)
        std_theta = np.std(chain_slice, axis=0)
        return mean_theta, std_theta, chain_slice

    def run_mcmc(self, n, theta_0, **kwargs):
        """!
        @brief Run the mcmc chain
        @param n int.  number of samples in chain to draw
        @param theta_0 np_1darray of initial guess
        """
        self._mcmc_run(n, theta_0, **kwargs)

    def _mcmc_run(self, *args, **kwarags):
        """! @brief Run the mcmc_chain.  Must be overridden."""
        raise NotImplementedError

    @property
    def chain(self):
        return self.am_chains[0]

    @property
    def acceptance_fraction(self):
        """!
        @brief Ratio of accepted samples vs total samples
        """
        return self.n_accepted / (self.n_accepted + self.n_rejected)

    @property
    def current_pos(self):
        return self.chain.current_pos()


def mh_kernel(mcmc_proposal, theta_chain, i=-1, verbose=0):
    """!
    @brief Metropolis-Hastings kernel.  Provides a new chain state
    given previous chain state and properties of a proposal density distribution.

    The kernel, \f[ K \f], maps the pevious chain state, \f[ x \f],
    to the new chain state, \f[ x' \f].  In matrix form:
    \f[
    K x = x'
    \f]
    In order for repeated application of an MCMC kernel to converge
    to the correct posterior distribution \f[\Pi()\f],
    it is sufficient but not neccissary to obey detailed balance:
    \f[
    K(x^i|x^{i-1})\Pi(x^{i}) = K(x^{i-1}|x^{i})\Pi(x^{i-1})
    \f]
    or
    \f[
    x_i K_{ij} = x_j K_{ji}
    \f]
    This means that a step in the chain is reversible.
    The goal in MCMC is to find \f[ K \f] that makes the state vector, \f[ x \f]
    become stationary at the desired distribution \f[ \Pi() \f]
    @param mcmc_proposl McmcProposal instance
    @param theta_chain  mcmc chain instance
    @param i  int. Index to use as current chain state
    @param verbose int or bool. default == 0 (optional)
    @return (np_1darray of new chain state, accepted int, rejected int)
    """
    assert isinstance(mcmc_proposal, McmcProposal)
    # set the gaussian proposal to be centered at current loc
    theta_current = theta_chain.current_pos()
    mcmc_proposal.mu = theta_current

    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)

    # propose a new place to go
    theta_prop = mcmc_proposal.sample_proposal()

    # compute acceptance ratio
    p_ratio = \
        mcmc_proposal.prob_ratio(
            theta_current,
            theta_prop)
    a_ratio = np.min((1, p_ratio))
    n_rejected, n_accepted = 0, 0
    if a_ratio >= 1.:
        # accept proposal, it is in area of higher prob density
        theta_new = theta_prop
        n_accepted = 1
        if verbose: print("Aratio: %f, Atest: %f , Accepted bc Aratio > 1" % (a_ratio, a_test))
    elif a_test < a_ratio:
        # accept proposal, even though it is "worse"
        theta_new = theta_prop
        n_accepted = 1
        if verbose: print("Aratio: %f, Atest: %f , Accepted by chance" % (a_ratio, a_test))
    else:
        # stay put, reject proposal
        theta_new = theta_current
        n_rejected = 1
        if verbose: print("Aratio: %f, Atest: %f , Rejected!" % (a_ratio, a_test))
    return theta_new, n_accepted, n_rejected


class Metropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    Proposal distribution is gaussian and symetric
    """
    def __init__(self, log_like_fn, ln_kwargs={}, **proposal_kwargs):
        proposal = 'Gauss'
        super(Metropolis, self).__init__(log_like_fn, ln_kwargs, proposal, **proposal_kwargs)

    def _init_chains(self, theta_0, **kwargs):
        self.am_chains = [McmcChain(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))]

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the metropolis algorithm.
        @param n  int. number of samples to draw.
        @param theta_0 np_1darray. initial guess for parameters.
        @param cov_est float or np_1darray.  Initial guess of anticipated theta variance.
            strongly recommended to specify, but is optional.
        """
        verbose = kwargs.get("verbose", 0)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0

        # initilize chain
        self._init_chains(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))

        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            theta_new, n_accepted, n_rejected = \
                mh_kernel(self.mcmc_proposal, self.chain, verbose=verbose)
            self.n_accepted += n_accepted
            self.n_rejected += n_rejected
            self.chain.append_sample(theta_new)


class AdaptiveMetropolis(Metropolis):
    """!
    @brief Adaptive Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    """
    def __init__(self, log_like_fn, ln_kwargs={}, **proposal_kwargs):
        proposal = 'Gauss'
        super(AdaptiveMetropolis, self).__init__(log_like_fn, ln_kwargs, proposal, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the adaptive metropolis algorithm.
        @param n  int. number of samples to draw.
        @param theta_0 np_1darray. initial guess for parameters.
        @param cov_est float or np_1darray.  Initial guess of anticipated theta variance.
            strongly recommended to specify, but is optional.
        @param adapt int.  Sample index at which to begin adaptively updating the
            proposal distribution (default == 200)
        @param lag  int.  Number of previous samples to use for proposal update (default == 100)
        @param lag_mod.  Number of iterations to wait between updates (default == 1)
        """
        verbose = kwargs.get("verbose", 0)
        adapt = kwargs.get("adapt", 1000)
        lag = kwargs.get("lag", 1000)
        lag_mod = kwargs.get("lag_mod", 100)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0

        # initilize chain
        self._init_chains(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))

        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            theta_new, n_accepted, n_rejected = \
                mh_kernel(self.mcmc_proposal, self.chain, verbose=verbose)
            self.chain.append_sample(theta_new)
            self.n_accepted += n_accepted
            self.n_rejected += n_rejected
            # continuously update the proposal distribution
            # if (lag > adapt):
            #    raise RuntimeError("lag must be smaller than adaptation start index")
            if i >= adapt and (i % lag_mod) == 0:
                if verbose: print("  Updating proposal cov at sample index = %d" % i)
                current_chain = self.chain[:i, :]
                self.mcmc_proposal.update_proposal_cov(current_chain[-lag:, :], verbose=verbose)


class DeMc(McmcSampler):
    """!
    @breif Differential-evolution metropolis sampler (DE-MC) note: not DREAM.
    Utilized multiple chains to crawl the parameter space more efficiently.
    At each MCMC iteration chains can be updated using a simple mutation operator:
    \f[
    \theta^* = \theta_i + \gamma(\theta_a + theta_b) + \varepsilon
    \f]

    TODO: Impl Sampler restarts.  Important for huge MCMC runs w/ crashes
    """
    def __init__(self, log_like_fn, n_chains=8, **proposal_kwargs):
        assert n_chains >= 4
        self.n_chains = n_chains
        proposal = 'Gauss'
        super(DeMc, self).__init__(log_like_fn, proposal, **proposal_kwargs)


    def _init_chains(self, theta_0, varepsilon=1e-6, **kwargs):
        # initilize chains
        self.am_chains = []
        for i in range(self.n_chains):
            self.am_chains.append(McmcChain(theta_0, varepsilon * kwargs.get("inflate", 1e1)))

    def _init_ln_like_fn(self, ln_kwargs={}):
        self._freeze_ln_like_fn(**ln_kwargs)

    def _mcmc_run(self, n, theta_0, varepsilon=1e-6, ln_kwargs={}, **kwargs):
        # self._freeze_ln_like_fn(**ln_kwargs)
        self._init_ln_like_fn(ln_kwargs)
        # params for DE-MC algo
        dim = len(theta_0)
        gamma = kwargs.get("gamma", 2.38 / np.sqrt(2. * dim))
        delayed_accept = kwargs.get("delayed_accept", True)

        # Init (serial) chains
        self._init_chains(theta_0, varepsilon, **kwargs)

        # DE-MC algo
        j = 0
        while j < (n - self.n_chains):
            banked_prop_array = []
            for i, current_chain in enumerate(self.am_chains):
                # randomly select chain pair from chain pool
                valid_pool_ids = np.delete(np.array(range(self.n_chains)), i)
                mut_chain_ids = np.random.choice(valid_pool_ids, replace=False, size=2)

                # get location of each mutation chain
                mut_a_chain = self.am_chains[mut_chain_ids[0]]
                mut_b_chain = self.am_chains[mut_chain_ids[1]]

                # generate proposal vector
                prop_vector = gamma * (mut_a_chain.current_pos - mut_b_chain.current_pos)
                prop_vector += current_chain.current_pos
                prop_vector += McmcChain.var_ball(varepsilon * 1e-3, dim)

                # note the probability ratio can be computed in parallel
                # accept or reject mutated chain loc
                alpha = self._mut_prop_ratio(self._frozen_ln_like_fn,
                                             current_chain.current_pos,
                                             prop_vector)
                if self.metropolis_accept(alpha):
                    new_state = prop_vector
                    self.n_accepted += 1
                else:
                    new_state = current_chain.current_pos
                    self.n_rejected += 1

                # imediately update chain[i] or bank updates untill all chains complete proposals
                if not delayed_accept:
                    current_chain.append_sample(new_state)
                else:
                    banked_prop_array.append(new_state)
                j += 1
            if delayed_accept:
                for i, current_chain in enumerate(self.am_chains):
                    current_chain.append_sample(banked_prop_array[i])


    def param_est(self, n_burn):
        chain_slice = self.super_chain[n_burn:, :]
        mean_theta = np.mean(chain_slice, axis=0)
        std_theta = np.std(chain_slice, axis=0)
        return mean_theta, std_theta, chain_slice

    @property
    def super_chain(self):
        return self._super_chain()

    def _super_chain(self):
        super_ch = np.zeros((self.n_chains * self.am_chains[0].chain.shape[0],
                             self.am_chains[0].chain.shape[1]))
        for i, chain in enumerate(self.am_chains):
            super_ch[i::len(self.am_chains), :] = chain.chain
        return super_ch

    def _mut_prop_ratio(self, log_like_fn, current_theta, mut_theta):
        # metropolis ratio
        alpha = np.min((1.0, np.exp(log_like_fn(mut_theta) - log_like_fn(current_theta))))
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha

    @staticmethod
    def metropolis_accept(alpha):
        return np.random.choice([True, False], p=[alpha, 1. - alpha])
