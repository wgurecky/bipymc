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
        dr_mcmc_theta_chain = McmcChain(theta_prop, varepsilon=1e-14)

        # run the delayed rejection chain
        dr_prop = dr_chain_step(dr_mcmc_prop, np.log(p_ratio), dr_mcmc_theta_chain)
        if dr_prop:
            n_accepted = 1
            n_rejected = 0
            theta_new = dr_prop
            print("Accepted DR chain prop!")
        else:
            n_rejected = 1
            n_accepted = 0
            theta_new = theta_current
    return theta_new, n_accepted, n_rejected


def dr_chain_step(mcmc_proposal, mh_ln_p_ratio, dr_theta_chain,
             current_depth=0, max_depth=5, verbose=0):
    """!
    @brief delayed rejection chain
    @param mcmc_proposal McmcProposal instance
    @param mh_ln_p_ratio natural log of mh proposal ratio
    @param dr_theta_chain McmcChain instance used to stored delayed
           rejection chain
    @param
    """
    theta_new = None
    # update the proposal distribution so that the mean
    # of the proposal is equal to the mean of the last n failed steps
    mcmc_proposal.mu = np.mean(dr_theta_chain.chain, axis=0)

    # shrink the proposal density cov
    mcmc_proposal.cov = mcmc_proposal.cov * 0.2

    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)

    # propose a new place to go
    dr_theta_prop = mcmc_proposal.sample_proposal()

    # compute new log of proposal prob ratio
    dr_p_ratio = mcmc_proposal.ln_prob_prop_ratio( \
            dr_theta_chain.current_pos, dr_theta_prop)

    # add new natural log of proposal ratio to past ln prop ratio
    print("==================")
    print("depth", current_depth)
    print("theta_0", dr_theta_chain.current_pos, "theta_new", dr_theta_prop)
    print("init_ln_p_ratio", mh_ln_p_ratio)
    dr_ln_p_ratio = mh_ln_p_ratio + dr_p_ratio

    # do standard metropolis accept/reject with updated ratio
    a_ratio = np.min((1, np.exp(dr_ln_p_ratio)))

    print("dr_ln_p_ratio", dr_ln_p_ratio)
    print("==================")

    if a_ratio >= 1.:
        theta_new = dr_theta_prop
        return theta_new
    elif a_test < a_ratio:
        theta_new = dr_theta_prop
        return theta_new
    elif current_depth >= max_depth:
        return None  # we failed all dealyed rejection attemts
    else:
        # recurse: step the dr chain again
        # recurse: run the dr chain
        current_depth += 1
        dr_theta_chain.append_sample(dr_theta_prop)
        dr_chain_step(mcmc_proposal, dr_ln_p_ratio, dr_theta_chain,
                      current_depth, max_depth)
    if theta_new:
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

        # initilize chain
        self._init_chains(theta_0, varepsilon=kwargs.get("varepsilon", 1e-12))

        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            theta_new, n_accepted, n_rejected = \
                dr_kernel(self.mcmc_proposal, self.chain, verbose=verbose)
            self.n_accepted += n_accepted
            self.n_rejected += n_rejected
            self.chain.append_sample(theta_new)
