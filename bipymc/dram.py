from samplers import *


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


