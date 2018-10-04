from __future__ import print_function, division
from six import iteritems
import numpy as np
import scipy.stats as stats
from copy import deepcopy


class McmcProposal(object):
    def __init__(self):
        pass

    def update_proposal_cov(self):
        raise NotImplementedError

    def sample_proposal(self, n_samples=1):
        raise NotImplementedError

    def prob_ratio(self):
        raise NotImplementedError


class GaussianProposal(McmcProposal):
    def __init__(self, mu=None, cov=None):
        """!
        @brief Init
        @param mu  np_1darray. centroid of multi-dim gauss
        @param cov np_ndarray. covariance matrix
        """
        self._cov_init = False
        self._mu = mu
        self._cov = cov
        super(GaussianProposal, self).__init__()

    def update_proposal_cov(self, past_samples, rescale=0, verbose=0):
        """!
        @brief fit gaussian proposal to past samples vector.

        The approximate optimal gaussian proposal distribution is:
        \f[
        \mathcal N(x, (2.38^2)\Sigma_n / d
        \f]
        (ref: http://probability.ca/jeff/ftpdir/adaptex.pdf)
        Where d is the dimension and \f$ \Sigma_n \f$ is the cov of the past samples
        in the chain at stage n.  Take x to be the current location of the
        chain.  In reality the proposal cov should be updated or
        adapted such that an
        acceptible acceptance ratio is reached... around 0.234
        """
        self._cov = np.cov(past_samples.T)
        # rescale cov matrix
        if rescale > 0:
            self._cov /= np.max(np.abs(self._cov)) / rescale
        if not self._cov.shape:
            self._cov = np.reshape(self._cov, (1, 1))
        # prevent possiblity of sigular cov matrix
        if not self._cov_init:
            self._cov += np.eye(self._cov.shape[0]) * 1e-8
            self._cov_init = True
        if verbose:
            print("New proposal cov = %s" % str(self._cov))

    def sample_proposal(self, n_samples=1):
        """!
        @brief Sample_proposal distribution
        """
        assert self.mu is not None
        assert self.cov is not None
        return np.random.multivariate_normal(self.mu, self.cov, size=n_samples)[0]

    def prob_ratio(self, ln_like_fn, theta_past, theta_proposed):
        """!
        @brief evaluate probability ratio:
        \f[
        \frac{\Pi(x^i)}{\Pi(x^{i-1})} \frac{g(x^{i-1}|x^i)}{g(x^i| x^{i-1})}
        \f]
        Where \f$ g() \f$ is the proposal distribution fn
        and \f[ \Pi \f] is the likelyhood function
        """
        return np.exp(self.ln_prob_ratio(ln_like_fn, theta_past, theta_proposed))

    def ln_prob_ratio(self, ln_like_fn, theta_past, theta_proposed):
        """!
        @brief evaluate log of probability ratio
        \f[
        ln(\frac{\Pi(x^i)}{\Pi(x^{i-1})} \frac{g(x^{i-1}|x^i)}{g(x^i| x^{i-1})})
        \f]
        Where \f$ g() \f$ is the proposal distribution fn
        and \f[ \Pi \f] is the likelyhood function.
        """
        assert self.mu is not None
        assert self.cov is not None
        g_ratio = lambda x_0, x_1: \
            stats.multivariate_normal.pdf(x_0, mean=theta_past, cov=self.cov) - \
            stats.multivariate_normal.pdf(x_1, mean=theta_proposed, cov=self.cov)
        g_r = g_ratio(theta_proposed, theta_past)  # should be 1 in symmetric case
        assert g_r == 0
        past_likelihood = ln_like_fn(theta_past)
        proposed_likelihood = ln_like_fn(theta_proposed)
        return proposed_likelihood - past_likelihood + g_r


    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov


class McmcSampler(object):
    """!
    @brief Markov Chain Monte Carlo (MCMC) base class.
    Contains common methods for dealing with the likelyhood function
    and for obtaining parameter estimates from the mcmc chain.
    """
    def __init__(self, log_like_fn, proposal, **proposal_kwargs):
        """!
        @brief setup mcmc sampler
        @param log_like_fn callable.  must return log likelyhood (float)
        @param proposal string.
        @param proposal_kwargs dict.
        """
        self.log_like_fn = log_like_fn
        if proposal == 'Gauss':
            self.mcmc_proposal = GaussianProposal(proposal_kwargs)
        else:
            raise RuntimeError("ERROR: proposal type: %s not supported." % str(proposal))
        self.chain = None
        self.n_accepted = 1
        self.n_rejected = 0
        self.chain = None

    def _freeze_ln_like_fn(self, **kwargs):
        """!
        @brief Freezes the likelyhood function.
        the log_like_fn should have signature:
            self.log_like_fn(theta, data=np.array([...]), **kwargs)
            and must return the log likelyhood
        """
        self._frozen_ln_like_fn = lambda theta: self.log_like_fn(theta, **kwargs)

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
    def acceptance_fraction(self):
        """!
        @brief Ratio of accepted samples vs total samples
        """
        return self.n_accepted / (self.n_accepted + self.n_rejected)

    @property
    def current_pos(self):
        return self.chain[-1, :]


def mh_kernel(i, mcmc_sampler, theta_chain, verbose=0):
    """!
    @brief Metropolis-Hastings mcmc kernel.
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
    @param i  int. Current chain index
    @param mcmc_sampler McmcSampler instance
    @param theta_chain  np_ndarray for sample storage
    @param verbose int or bool. default == 0 (optional)
    """
    theta = theta_chain[i, :]
    # set the gaussian proposal to be centered at current loc
    mcmc_sampler.mcmc_proposal.mu = theta
    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)
    # propose a new place to go
    theta_prop = mcmc_sampler.mcmc_proposal.sample_proposal()
    # compute acceptance ratio
    a_ratio = np.min((1, mcmc_sampler.mcmc_proposal.prob_ratio(
        mcmc_sampler._frozen_ln_like_fn,
        theta,
        theta_prop)))
    if a_ratio >= 1.:
        # accept proposal, it is in area of higher prob density
        theta_chain[i+1, :] = theta_prop
        mcmc_sampler.n_accepted += 1
        if verbose: print("Aratio: %f, Atest: %f , Accepted bc Aratio > 1" % (a_ratio, a_test))
    elif a_test < a_ratio:
        # accept proposal, even though it is "worse"
        theta_chain[i+1, :] = theta_prop
        mcmc_sampler.n_accepted += 1
        if verbose: print("Aratio: %f, Atest: %f , Accepted by chance" % (a_ratio, a_test))
    else:
        # stay put, reject proposal
        theta_chain[i+1, :] = theta
        mcmc_sampler.n_rejected += 1
        if verbose: print("Aratio: %f, Atest: %f , Rejected!" % (a_ratio, a_test))
    return theta_chain


class Metropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    Proposal distribution is gaussian and symetric
    """
    def __init__(self, log_like_fn, **proposal_kwargs):
        proposal = 'Gauss'
        super(Metropolis, self).__init__(log_like_fn, proposal, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, ln_kwargs={}, **kwargs):
        """!
        @brief Run the metropolis algorithm.
        @param n  int. number of samples to draw.
        @param theta_0 np_1darray. initial guess for parameters.
        @param cov_est float or np_1darray.  Initial guess of anticipated theta variance.
            strongly recommended to specify, but is optional.
        """
        verbose = kwargs.get("verbose", 0)
        self._freeze_ln_like_fn(**ln_kwargs)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0
        theta_chain = np.zeros((n, np.size(theta_0)))
        self.chain = theta_chain
        theta_chain[0, :] = theta_0
        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            mh_kernel(i, self, theta_chain, verbose=verbose)
        self.chain = theta_chain


class AdaptiveMetropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    """
    def __init__(self, log_like_fn, **proposal_kwargs):
        proposal = 'Gauss'
        super(AdaptiveMetropolis, self).__init__(log_like_fn, proposal, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, ln_kwargs={}, **kwargs):
        """!
        @brief Run the metropolis algorithm.
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
        self._freeze_ln_like_fn(**ln_kwargs)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0
        theta_chain = np.zeros((n, np.size(theta_0)))
        self.chain = theta_chain
        theta_chain[0, :] = theta_0
        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            mh_kernel(i, self, theta_chain, verbose=verbose)
            # continuously update the proposal distribution
            # if (lag > adapt):
            #    raise RuntimeError("lag must be smaller than adaptation start index")
            if i >= adapt and (i % lag_mod) == 0:
                if verbose: print("  Updating proposal cov at sample index = %d" % i)
                current_chain = theta_chain[:i, :]
                self.mcmc_proposal.update_proposal_cov(current_chain[-lag:, :], verbose=verbose)
        self.chain = theta_chain


class McmcChain(object):
    """!
    @brief A simple markov chain with some helper functions.
    """
    def __init__(self, theta_0, varepsilon=1e-6, global_id=0, mpi_comm=None, mpi_rank=None):
        self.global_id = global_id
        # self.mpi_comm, self.mpi_rank = mpi_comm, mpi_rank
        self._dim = len(theta_0)
        theta_0 = np.asarray(theta_0) + self.var_ball(varepsilon, self._dim)
        # init the 2d array of shape (iteration, <theta_vec>)
        self._chain = np.array([theta_0])

    @staticmethod
    def var_ball(varepsilon, dim):
        # tight gaussian ball
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

    def auto_corr(self, lag):
        """!
        @brief Compute auto correlation in the chain.
        """
        pass

    @property
    def chain(self):
        return self._chain

    @property
    def current_pos(self):
        return self.chain[-1, :]

    @property
    def dim(self):
        return self._dim


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
