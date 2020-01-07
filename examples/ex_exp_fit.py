from __future__ import print_function, division
import numpy as np
import sys
import scipy.stats as stats
from scipy.io import loadmat
from scipy.optimize import curve_fit
from mpi4py import MPI
from emcee import EnsembleSampler
import corner
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
# import seaborn as sns
# sns.set_style("whitegrid")
import matplotlib as mpl
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
# mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
comm = MPI.COMM_WORLD
try:
    from bipymc.samplers import DeMc, AdaptiveMetropolis, Metropolis
    from bipymc.demc import DeMcMpi
    from bipymc.dream import DreamMpi
    from bipymc.gp.bayes_opti import bo_optimizer
    from bipymc.mc_plot import mc_plot
except:
    # add to path
    sys.path.append('../.')
    from bipymc.samplers import DeMc, AdaptiveMetropolis, Metropolis
    from bipymc.demc import DeMcMpi
    from bipymc.dream import DreamMpi
    from bipymc.gp.bayes_opti import bo_optimizer
    from bipymc.mc_plot import mc_plot
np.random.seed(42)



def exp_c1_model_full(tau, c_inf, c_0, leak, t, **kwargs):
    """!
    @brief Full model we want to fit to the data
    """
    v2 = -1.0
    return c_inf + c_0 * v2 * np.exp(-t / tau) - leak * t


def exp_c1_model(tau, c_inf, t, **kwargs):
    """!
    @brief Simplified model without leakage
    unknowns are tau and c_inv,
    @param tau model param
    @param c_inf model param
    @param t is time vector
    """
    v1, v2 = 1.0, 1.0
    return c_inf * np.exp(-t / tau)

def model(theta, t):
    """!
    @brief model liklihood.  This is proportional to the
    true liklihood function.
    """
    tau, c_inf, c_0, leak, sigma = theta
    return exp_c1_model_full(tau, c_inf, c_0, leak, t)

def model_se(theta, t, y_data):
    """!
    @brief model squared error
    """
    tau, c_inf, c_0, leak = theta
    m = exp_c1_model_full(tau, c_inf, c_0, leak, t)
    return np.sum((m - y_data) ** 2.)

def ln_model_like(theta, t, y_data):
    """!
    @brief model log-liklihood
    @param theta list of model params
    """
    # y_sigma = 1.0e-8  # y_errs
    y_sigma = theta[-1]
    ln_model = np.sum((model(theta, t) - y_data) ** 2. / y_sigma - np.log(1.0 / y_sigma))
    # ln_model = np.sum((model(theta, t) - y_data) ** 2.)
    return -0.5 * (ln_model)

def lnprob(theta, t, y_data):
    """!
    @brief log prob of RHS of bayes
    \f[
    lnprob = ln(p(theta|data)p(theta))
           = ln(p(theta|data) + ln(p(theta))
    \f]
    where p(theta) is the prior distribution
    and p(theta|data) is the likelihood function of the model
    @param theta list of parameters:
        theta[0] == tau
        theta[1] == c_inf
    @param t time vector
    @param y_data experimental concentration data
    """
    lp = ln_params_prior(*theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_model_like(theta, t, y_data)

def ln_params_prior(tau, c_inf, c_0, leak, sigma):
    """!
    @brief Prior distributions for model params tau and c_inf
    @use flat priors
    """
    sigma_range = np.array([0, 1.0])
    c_0_range = np.array([-1.0, 1.])
    leak_range = np.array([-5., 5.0])
    c_inf_guess_range = np.array([-5, 5.0])
    tau_guess_range = np.array([1.0, 50])
    if (c_inf_guess_range[0] < c_inf < c_inf_guess_range[1]) \
        and (tau_guess_range[0] < tau < tau_guess_range[1]) \
        and (c_0_range[0] < c_0 < c_0_range[1]) \
        and (leak_range[0] < leak < leak_range[1]) \
        and (sigma_range[0] < sigma < sigma_range[1]):
        return 0.0
    else:
        return -np.inf


def fit_exp_data(theta_0, mcmc_algo="DE-MC"):
    """!
    @brief Fit an exponential model to some data
    """

    # get data
    t_data, y_data = read_data()

    sigma_0 = 1e-4
    theta_0 = np.array(list(theta_0) + [sigma_0])

    # Run MCMC
    #my_mcmc = DeMcMpi(lnprob, theta_0, n_chains=comm.size*10, mpi_comm=comm,
    #             varepsilon=1e-9, inflate=1e-1, ln_kwargs={'y_data': y_data, 't': t_data})
    my_mcmc = DreamMpi(lnprob, theta_0, n_chains=comm.size*20, mpi_comm=comm,
                 varepsilon=1e-8, inflate=1e-1, ln_kwargs={'y_data': y_data, 't': t_data})
    my_mcmc.run_mcmc(1000 * 100, suffle=True, flip=0.5)

    # Run MCMC
    # my_mcmc = DeMc(lnprob, n_chains=comm.size*10, mpi_comm=comm,
    #             ln_kwargs={'y_data': y_data, 't': t_data})
    # my_mcmc.run_mcmc(2000 * 100, theta_0, varepsilon=1e-9, inflate=1e-1)

    # Run Emcee MCMC
    if comm.rank == 0:
        ndim, nwalkers = 5, 100
        pos = [theta_0 + 1e-6*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(t_data, y_data))
        sampler.run_mcmc(pos, 1000)  # 100 * 1000 tot samples
        samples = sampler.chain[:, 400:, :].reshape((-1, ndim))
        fig = corner.corner(samples, labels=["$\tau$", "$c_\infty$", "$c_0$", "l", r"$\sigma$"])
        fig.savefig("exp_emcee_out.png")
        print("=== EMCEE ===")
        print("Emcee mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))

    # view results
    print("=== Opti values by Bipymc MCMC ===")
    print("[tau, c_inf, c_0, leakage]:")
    theta_est, sig_est, chain = my_mcmc.param_est(n_burn=400 * 100)
    theta_est_, sig_est_, full_chain = my_mcmc.param_est(n_burn=0)
    if comm.rank == 0:
        print("MCMC Esimated params: %s" % str(theta_est))
        print("MCMC Estimated params sigma: %s " % str(sig_est))
        print("Acceptance fraction: %f" % my_mcmc.acceptance_fraction)
        print("P_cr: %s" % str(my_mcmc.p_cr))
        # vis the parameter estimates
        mc_plot.plot_mcmc_params(chain,
                labels=[r"$\tau$", "$c_\infty$", "$c_0$", "leak", r"$\sigma$"],
                savefig='exp_mcmc_out.png',)
        # vis the full chain
        mc_plot.plot_mcmc_chain(full_chain,
                labels=[r"$\tau$", "$c_\infty$", "$c_0$", "leak", r"$\sigma$"],
                savefig='exp_chain_out.png',)

        # plot trajectories
        xdata, ydata = read_data()
        plt.close()
        i=0
        reduced_model = lambda t, tau, c_inf, c_0, set_leak, sigma: exp_c1_model_full(tau, c_inf, c_0, set_leak, t)
        for sample in chain:
            i+=1
            plt.plot(xdata, np.abs(reduced_model(xdata, *sample) - sample[1]) / sample[1], lw=0.2, alpha=0.02, c='b')
            if i > 1000:
                break
        plt.title("MCMC Fit")
        plt.plot(xdata, np.abs(reduced_model(xdata, *theta_est) - theta_est[1]) / theta_est[1], lw=1.0, alpha=0.8, label="MCMC", c='b')
        plt.scatter(xdata, np.abs(ydata - theta_est[1]) / theta_est[1], s=2, alpha=0.9, c='r', label="data")
        plt.ylabel("$|C_t - c_\infty| /C_\infty$")
        plt.xlabel("time")
        plt.legend()
        plt.savefig("mcmc_trajectories.png", dpi=160)

        # plot the slopes at the end
        pass

        # compute and plot c_o/c_inf
        c_inf_samples = chain[:, 1]
        c_0_samples = chain[:, 2]
        c_ratio = c_0_samples / c_inf_samples
        c_ratio_avg, c_ratio_sd = np.mean(c_ratio), np.std(c_ratio)
        print("c_0/c_inf estimate: %0.3e +/- %0.3e" % (c_ratio_avg, c_ratio_sd))


def gen_initial_guess():
    # use scipy fit to generate an initial guess for the model params
    # set_leak = 0
    reduced_model = lambda t, tau, c_inf, c_0, set_leak: exp_c1_model_full(tau, c_inf, c_0, set_leak, t)
    tau_0 = 28.
    c_inf_0 = 0.381
    c_0_0 = -0.6
    leak_0 = -4.932e-4
    p0 = [tau_0, c_inf_0, c_0_0, leak_0]
    xdata, ydata = read_data()
    popt, pcov = curve_fit(reduced_model, xdata, ydata, p0)
    print("=== Opti values by scipy curve_fit ===")
    print("[tau, c_inf, c_0, leakage]:")
    print(popt)
    plt.scatter(xdata, np.abs(ydata - popt[1]) / popt[1], s=2, alpha=0.5, c='r', label="data")
    plt.plot(xdata, np.abs(reduced_model(xdata, *popt) - popt[1]) / popt[1], label="LS fit")
    plt.legend()
    plt.title("Least Squares")
    plt.xlabel("time")
    plt.ylabel("$|C_t - c_\infty| /C_\infty$")
    plt.savefig("initial_fit.png", dpi=160)
    plt.close()
    return popt


def gen_initial_guess_bo():
    # use bayesian optimization to generate an initial guess for the model params
    xdata, ydata = read_data()
    r = []
    def model_resid(Xtest):
        r = []
        for theta in Xtest:
            r.append(model_se(theta, xdata, ydata))
        return np.array(r).flatten()
    my_bounds = ((10, 50), (0.1, 1.0), (-1.0, 1.0), (-1e-3, 0.0))
    my_bo = bo_optimizer(model_resid, dim=4, s_bounds=my_bounds, n_init=2, y_sigma=1e-6,
            gp_fit_kwargs={'maxf': 1000})
    popt = my_bo.optimize(40, n_samples=2000, max_depth=3, diag_scale=1e-6, mode='min')

    print("=== Opti values by bayesian opt ===")
    print("[tau, c_inf, c_0, leakage]:")
    print(popt)
    return popt

def read_data(file_name = 'concentration_data.mat', drop=10, col=3):
    """!
    @brief read raw data in
    @param drop  drop the first n datapoints
    """
    # read in experimental data from files
    raw_data = loadmat(file_name)
    times, c_t = raw_data['time'].flatten()[drop:], raw_data['dummy'][col].flatten()[drop:]
    return times, c_t

if __name__ == "__main__":
    #popt = gen_initial_guess()
    popt_bo = gen_initial_guess_bo()
    fit_exp_data(popt_bo, "DE-MC")
