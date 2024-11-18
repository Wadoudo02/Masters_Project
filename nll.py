import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import find_crossings

def TwoDeltaNLL(x):
    x = np.array(x)
    return 2*(x-x.min())

#Takes optional param cat which if provided only get NLL over that category
def calc_NLL(hists, mus, conf_matrix = [],signal='ttH'):
    NLL_vals = []
    # Loop over recon categories
    #print(hists)
    for cat, yields in hists.items():
        n_bins = len(list(yields.values())[0])
        e = np.zeros(n_bins)
        n = np.zeros(n_bins)
        #Loop over prod modes, truth bins so stuff inside log
        for proc, bin_yields in yields.items():
            if proc == signal:
                #Case where conf matrix provided
                if len(conf_matrix)!=0:
                    '''
                    For the particular recon category "cat" we sum over each truth category * the mu belonging to that category
                    which in this case is 1 for all categories other than the one we are looking at for which it is mu, as we 
                    are performing a frozen fit.
                    '''
                    for truth_cat in range(5):
                        e+=mus[truth_cat]*hists[truth_cat][signal]*conf_matrix[cat][truth_cat]
                else:
                    e+=mus[0]*bin_yields
                
            else:
                e += bin_yields
            n += bin_yields
        nll = e-n*np.log(e)
        NLL_vals.append(nll)
    #print(NLL_vals, np.array(NLL_vals).sum())
    return np.array(NLL_vals).sum()

#Taken from Wadoud
def calc_NLL_comb(combined_histogram, mus, signal='ttH'):
    """
    Calculate the NLL using the combined 25-bin histogram with variable \mu parameters.

    Parameters:
        combined_histogram (dict): Combined histogram for each process across 25 bins.
        mus (list or array): Signal strength modifiers, one for each truth category.
        conf_matrix (ndarray): Confusion matrix for adjusting expected yields.
        signal (str): The signal process (default 'ttH').

    Returns:
        float: Total NLL.
    """
    NLL_total = 0.0
    num_bins = len(next(iter(combined_histogram.values())))  # Total bins (should be 25)

    # Loop over each bin in the combined histogram
    for bin_idx in range(num_bins):
        expected_total = 0.0
        observed_count = 0.0

        for proc, yields in combined_histogram.items():
            if signal in proc:
                # Extract the truth category index from the signal label, e.g., "ttH_0"
                truth_cat_idx = int(proc.split('_')[1])
                #print(mus, truth_cat_idx)
                mu = mus[truth_cat_idx]  # Apply the appropriate \mu for this truth category
                expected_total += mu * yields[bin_idx]
            else:
                expected_total += yields[bin_idx]

            observed_count += yields[bin_idx]  # Observed count from all processes in this bin
        
        # Calculate NLL contribution for this bin
        NLL_total += observed_count * np.log(expected_total) - expected_total

    return -NLL_total

# Scans over 1 mu index and at each step minimises all others so gives best fit for that mu value
def profiled_NLL_fit(combined_histogram, conf_matrix, mu_idx):
    num_truth = len(conf_matrix[0])

    mu_vals = np.linspace(0, 3, 100)
    nll = []

    def calc_nll_fixed_mu(fixed, others):
        mus = np.array(others)
        mus = np.insert(mus, mu_idx, fixed)
        return calc_NLL_comb(combined_histogram, mus)
    
    for mu in mu_vals:
        #Generating ones for all but 1 mu
        guess = np.ones(num_truth-1)

        res = minimize(
            lambda other_mus: calc_nll_fixed_mu(mu, other_mus),
            guess,
            method="L-BFGS-B"
        )

        nll.append(res.fun)
    
    vals = find_crossings((mu_vals, TwoDeltaNLL(nll)), 1.)
    return (vals[0], nll)

def global_nll_fit(combined_histogram, conf_matrix):
    num_truth = len(conf_matrix[0])

    def calc_nll_mus_only(mus):
        return calc_NLL_comb(combined_histogram, mus)
    guess = np.ones(num_truth)
    res = minimize(
        calc_nll_mus_only,
        guess,
        method="L-BFGS-B"
    )
    hessian_inv = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
    hessian = np.linalg.inv(hessian_inv)
    return (res.x, res.fun, hessian_inv, hessian)