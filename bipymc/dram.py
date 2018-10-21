from samplers import *
from chain import *
from copy import deepcopy
import numpy as np


def dr_kernel(mcmc_proposal, theta_chain, i=-1, verbose=0):
    """!
    @brief Delayed rejection kenel
    """
    depth = 0
    assert isinstance(mcmc_proposal, McmcProposal)
    theta_current = theta_chain.current_pos
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
    n_dr_accept = 0
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
        theta_new = theta_current
        n_rejected = 1
        if verbose: print("Aratio: %f, Atest: %f , Rejected!" % (a_ratio, a_test))

        # TODO: remove deep copies
        dr_mcmc_prop = deepcopy(mcmc_proposal)
        # make a new dr chain starting with the failed proposal as "initial guess"
        # TODO: varepsilon should be 0 here
        dr_mcmc_theta_chain = McmcChain(theta_prop, varepsilon=1e-20)

        # run the delayed rejection chain
        dr_prop = dr_chain_step(dr_mcmc_prop, dr_mcmc_theta_chain, theta_current, a_ratio)
        if dr_prop is not None:
            n_accepted = 1
            n_rejected = 0
            n_dr_accept += 1
            theta_new = dr_prop
            # print("Accepted DR chain prop!")
        else:
            n_rejected = 1
            n_accepted = 0
            theta_new = theta_current
    return theta_new, n_accepted, n_rejected, n_dr_accept


def dr_accept_prob(mcmc_proposal, theta_a, theta_b):
    p_ratio = mcmc_proposal.prob_ratio(theta_a, theta_b)
    return np.min((1, p_ratio))


def dr_chain_step(mcmc_proposal, dr_theta_chain, theta_0, alpha_old,
             current_depth=0, max_depth=0, verbose=0):
    """!
    @brief delayed rejection chain
    @param mcmc_proposal McmcProposal instance
    @param dr_theta_chain McmcChain instance used to stored delayed
           rejection chain
    @param
    """
    theta_new = None
    # update the proposal distribution so that the mean
    # of the proposal is equal to the mean of the last n failed steps
    mcmc_proposal.mu = np.mean(dr_theta_chain.chain, axis=0)

    # shrink the proposal density cov
    mcmc_proposal.cov = mcmc_proposal.cov * 0.4

    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)

    # propose a new place to go
    dr_theta_prop = mcmc_proposal.sample_proposal()

    # compute new log of proposal prob ratio
    ln_p_ratio = mcmc_proposal.ln_prob_prop_ratio(theta_0, dr_theta_prop)

    # compute new alpha
    alpha_new = dr_accept_prob(mcmc_proposal, dr_theta_prop, dr_theta_chain.current_pos)

    # update delayed rejection alpha
    dr_p_accept = np.min((1., np.exp( ln_p_ratio + np.log(1. - alpha_new) - np.log(1. - alpha_old) ) ))

    if dr_p_accept >= 1.:
        theta_new = dr_theta_prop
        return theta_new
    elif a_test < dr_p_accept:
        theta_new = dr_theta_prop
        return theta_new
    elif current_depth >= max_depth:
        return None  # we failed all dealyed rejection attemts
    else:
        # recurse: step the dr chain again
        current_depth += 1
        dr_theta_chain.append_sample(dr_theta_prop)
        dr_chain_step(mcmc_proposal, dr_theta_chain, theta_0, dr_p_accept,
                      current_depth, max_depth)
    if theta_new is not None:
        return theta_new
    else:
        return None


class DrMetropolis(Metropolis):
    """!
    @brief Delayed rejection Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    Proposal distribution is gaussian and symetric
    """
    def __init__(self, log_like_fn, ln_kwargs={}, **proposal_kwargs):
        super(DrMetropolis, self).__init__(log_like_fn, ln_kwargs, **proposal_kwargs)

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
        self.n_dr_accept = 0

        # initilize chain
        self._init_chains(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))

        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            theta_new, n_accepted, n_rejected, n_dr_accept = \
                dr_kernel(self.mcmc_proposal, self.chain, verbose=verbose)
            self.n_accepted += n_accepted
            self.n_rejected += n_rejected
            self.n_dr_accept += n_dr_accept
            self.chain.append_sample(theta_new)
        print("n dr accept", self.n_dr_accept)


class Dram(Metropolis):
    """!
    @brief Delayed rejection Adaptive Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    """
    def __init__(self, log_like_fn, ln_kwargs={}, **proposal_kwargs):
        super(Dram, self).__init__(log_like_fn, ln_kwargs, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the delayed rejection adaptive metropolis algorithm.
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
        self.n_dr_accept = 0

        # initilize chain
        self._init_chains(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))

        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            theta_new, n_accepted, n_rejected, n_dr_accept = \
                dr_kernel(self.mcmc_proposal, self.chain, verbose=verbose)
            self.chain.append_sample(theta_new)
            self.n_dr_accept += n_dr_accept
            self.n_accepted += n_accepted
            self.n_rejected += n_rejected
            # continuously update the proposal distribution
            # if (lag > adapt):
            #    raise RuntimeError("lag must be smaller than adaptation start index")
            if i >= adapt and (i % lag_mod) == 0:
                if verbose: print("  Updating proposal cov at sample index = %d" % i)
                current_chain = self.chain.chain[:i, :]
                self.mcmc_proposal.update_proposal_cov(current_chain[-lag:, :], verbose=verbose)
        print("n dr accept", self.n_dr_accept)
