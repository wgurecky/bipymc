from __future__ import division, print_function
import numpy as np
import unittest
from bipymc.gp.gpr import gp_regressor
np.random.seed(42)


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_1d_gp(self):
        # sin data with noise
        Xtrain = np.random.uniform(-4, 4, 60).reshape(-1,1)
        ytrain = np.sin(Xtrain) + np.random.uniform(-1e-2, 1e-2, Xtrain.size).reshape(Xtrain.shape)

        # fit
        my_gpr = gp_regressor(domain_bounds=((-4., 4.),))
        my_gpr.fit(Xtrain, ytrain, y_sigma=1e-2)

        # eval at test points
        n = 500
        Xtest = np.linspace(np.min(Xtrain), np.max(Xtrain), n).reshape(-1,1)
        ytest_mean = my_gpr.predict(Xtest)
        ytest_samples = my_gpr.sample_y(Xtest, n_draws=200, chunk_size=1e10)
        ytest_sd = my_gpr.predict_sd(Xtest)

        # check rms
        diffs = np.sin(Xtest) - ytest_mean
        rms = np.sqrt(np.sum((diffs) ** 2.0) / (ytest_mean.size))
        self.assertLess(rms, 1e-2)
        l1_norm = np.max(np.abs(diffs))
        self.assertLess(l1_norm, 1e-1)

    def test_2d_gp(self):
        # 2d quadradic gp fit test
        def obj_fn_2d(Xtest):
            X, Y = Xtest[:, 0], Xtest[:, 1]
            return X ** 2.0 + Y ** 2.0

        x1 = np.random.uniform(-4, 4, 120)
        y1 = np.random.uniform(-4, 4, 120)
        Xtrain = np.array([x1, y1]).T
        Z = obj_fn_2d(Xtrain)

        my_gpr_2d = gp_regressor(ndim=2, domain_bounds=((-4., 4.), (-4., 4.),))
        my_gpr_2d.fit(Xtrain, Z, y_sigma=1e-4)

        n = 100
        Xtest = np.linspace(-4.0, 4.0, n)
        xt, yt = np.meshgrid(Xtest, Xtest)
        Xtest = np.array((xt.flatten(), yt.flatten())).T
        ytest_mean = my_gpr_2d.predict(Xtest)
        zt = ytest_mean.reshape(xt.shape)

        # check rms
        diffs = obj_fn_2d(Xtest) - ytest_mean
        rms = np.sqrt(np.sum((diffs) ** 2.0) / (ytest_mean.size))
        self.assertLess(rms, 1e-1)
        l1_norm = np.max(np.abs(diffs))
        self.assertLess(l1_norm, 5e-1)
