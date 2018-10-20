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
    # update the proposal distribution so that the mean
    # of the proposal is equal to the mean of the last n failed steps
    mcmc_proposal.mu = np.mean(dr_theta_chain.chain, axis=0)
    # TODO: update covarience of proposal too?

    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)

    # propose a new place to go
    dr_theta_prop = mcmc_proposal.sample_proposal()

    # compute new log of proposal prob ratio
    dr_p_ratio = mcmc_proposal.ln_prob_prop_ratio( \
            dr_theta_chain.current_pos, dr_theta_prop)

    # add new natural log of proposal ratio to past ln prop ratio
    dr_ln_p_ratio = mh_ln_p_ratio + dr_p_ratio

    # do standard metropolis accept/reject with updated ratio
    a_ratio = np.min((1, np.exp(dr_ln_p_ratio)))

    if a_ratio >= 1.:
        theta_new = theta_prop
        return theta_new
    elif a_test < a_ratio:
        theta_new = theta_prop
        return theta_new
    elif current_depth >= max_depth:
        return None  # we failed all dealyed rejection attemts
    else:
        current_depth += 1
        dr_theta_chain.append_sample(dr_theta_prop)
        # recurse: run the dr chain
        dr_chain(mcmc_proposal, dr_ln_p_ratio, dr_theta_chain,
                 current_depth, max_depth)



class DrChain(object):
    """!
    @brief Delayed rejection chain.
    """
    def __init__(self, ln_like_fn, **kwargs):
        self.current_depth = 1
        self.dr_sample_chain = []
        self.dr_prop_chain = []
        self.max_depth = kwargs.get("max_depth", 5)
        self.ln_like_fn = ln_like_fn

    def run_dr_chain(self, dr_seed_chain, prop_base_cov):
        """!
        Run the delayed rejection (DR) chain.  break
        early if we happen to accept a sample.
        @prop_base_cov  covarience of the base MCMC proposal
        """
        assert len(dr_seed_chain) == 2
        #param lambda_0 original location of MCMC chain before DR
        #param beta_0 proposed, but rejected sample from original MCMC chain before DR
        lambda_0 = dr_seed_chain[0]
        beta_1 = dr_seed_chain[1]
        self.dr_sample_chain = deepcopy(dr_seed_chain)

        for i in range(self.max_depth):
            depth = i + 1
            # get a proposal density distribution based on all previous
            # samples in the dr chain.
            self.dr_prop_chain.append(DrGaussianProposal(self.ln_like_fn,
                                      prop_base_cov))
            new_prop_dist = self.dr_prop_chain[-1]._wrapped_mv_gausasin(dr_sample_chain)
            # sample the proposal density distribution
        pass

class DrGaussianProposal(McmcProposal):
    """!
    @brief Delayed rejection gaussian proposal chain
    with proposal shrinkage.  The proposal density of a DR chain
    is updated with knowlege of other past proposed steps in the DR chain.
    """
    def __init__(self, ln_like_fn, base_cov):
        self.ln_like_fn = ln_like_fn
        self.depth = 1
        self.cov_base = base_cov
        super(DrGaussianProposal, self).__init__()

    def multi_stage_proposal_dist(self, prop_lambda, prop_past_beta):
        """!
        @brief Compute
        \beta_i \sim g_i(\beta_i | \lambda, \beta_1, ... \beta_{i-1})
        """
        pass

    def prob_ratio(self, *args, **kwargs):
        return np.exp(self.ln_prob_ratio(*args, **kwargs))

    def ln_prob_ratio(self, *args, **kwargs):
        pass

    def _ln_likelihood_ratio(self, beta_past, beta_proposed):
        past_likelihood = self.ln_like_fn(beta_past)
        proposed_likelihood = self.ln_like_fn(beta_proposed)
        return proposed_likelihood - past_likelihood

    def _ln_proposal_ratio(self, beta_proposal, beta_dr_list):
        beta_dr_list= []
        numerator = ()
        pass

    def _wrapped_mv_gausasin(self, beta_dr_list):
        shrinkage_array = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        mu = np.mean(beta_dr_list)
        cov = self.cov_base  * shrinkage_array[len(beta_dr_list)]
        return stats.multivariate_normal(mean=mu, cov=cov)

    def _sample_wrapped_mv_gaussian(self, n):
        pass

    def _ln_acceptance_ratio(self):
        pass


