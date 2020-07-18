from __future__ import print_function, division
import numpy as np


def var_ball(varepsilon, dim):
    """!
    @brief Draw single sample from tight gaussian ball
    @param varepsilon  float or 1d_array of len dim
    @param dim  dimension of gaussian ball
    """
    eps = 0.
    if np.all(np.asarray(varepsilon) > 0):
        eps = np.random.multivariate_normal(np.zeros(dim),
                                            np.eye(dim) * np.asarray(varepsilon),
                                            size=1)[0]
    return eps

def var_box(varepsilon, dim):
    """!
    @brief Draw single sample from tight uniform box
    @param varepsilon  float or 1d_array of len dim
    @param dim  dimension of uniform distribution
    """
    eps = 0.
    if np.all(np.asarray(varepsilon) > 0):
        eps = np.random.uniform(low=-np.asarray(varepsilon) * np.ones(dim),
                                high=np.asarray(varepsilon) * np.ones(dim))
    return eps
