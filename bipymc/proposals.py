from __future__ import print_function, division
from six import iteritems
import numpy as np
import scipy.stats as stats


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


