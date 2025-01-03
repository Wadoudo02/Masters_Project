import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.integrate import quad
import json

from utils import *


plot_entire_chain = False

plot_fraction = False

# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF x 10", "red"],
    "VH" : ["VH x 10", "orange"],
    #"Data" : ["Data", "green"]
}

plot_size = (12, 8)

# Load dataframes
dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]
    '''
    if proc == "Data":
        dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']
        
        bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
        labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
        dfs[proc]['category'] = pd.cut(dfs[proc]['pt_sel'], bins=bins, labels=labels, right=False)
        
        continue
    '''

    # Reweight to target lumi
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)

    # Calculate true weight: remove x10 multiplier for signal
    if proc in ['ggH', 'VBF', 'VH', 'ttH']:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']/10
    else:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']
        

    # Add variables
    # Example: (second-)max-b-tag score
    b_tag_scores = np.array(dfs[proc][['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    
    
    # Add nans back in for plotting tools below
    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score_sel'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score_sel'] = second_max_b_tag_score
    
    # Apply selection: separate ttH from backgrounds + other H production modes
    yield_before_sel = dfs[proc]['true_weight'].sum()
    
    
    mask = dfs[proc]['n_jets_sel'] >= 2
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.7)
    mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    # Categorise events: separate regions of high EFT enhancement vs low EFT enhancement

    # Categorise events: separate into 5 categories by pt
    bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
    labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
    dfs[proc]['category'] = pd.cut(dfs[proc]['pt_sel'], bins=bins, labels=labels, right=False)
    
# Extract different cat integers
cats_unique = []
for proc in procs.keys():
    for cat in np.unique(dfs[proc]['category']):
        if cat not in cats_unique:
            cats_unique.append(cat)
  
cats_unique = labels.copy()

def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 100))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category

background_estimates = {}

mass_range = (120, 130)  # Needed here to get BG estimate
mass_bins = 5

v = "mass"
v_dfs = v + "_sel"
for cat in cats_unique:
    print(f" --> Processing: {v} in category {cat}")
    nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]

    # Loop over procs and add histogram
    for proc in procs.keys():
        label, color = procs[proc]

        cat_mask = dfs[proc]['category'] == cat

        x = np.array(dfs[proc][v_dfs][cat_mask])

        # Event weight
        w = np.array(dfs[proc]['plot_weight'])[cat_mask]

        counts, bin_edges = np.histogram(x, bins=nbins, range=xrange, weights=w)

        if proc == "background":
           # breakpoint()
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # Perform curve fitting, ignoring empty bins (where counts == 0)
            non_zero_indices = counts > 0
            popt, pcov = curve_fit(exponential_decay, bin_centers[non_zero_indices], counts[non_zero_indices])
            A, lambd = popt  # Unpack fitted parameters

            # Background estimate
            BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
            bin_estimates = []
            for i in range(len(BG_estimate_bin_edges) - 1):
                integral, _ = quad(exponential_decay, BG_estimate_bin_edges[i], BG_estimate_bin_edges[i + 1], args=(A, lambd))
                bin_estimates.append(integral)

            print(f"Background estimates for category {cat}: {bin_estimates}")

            # Store the result
            if cat not in background_estimates:
                background_estimates[cat] = {}
            background_estimates[cat][proc] = bin_estimates

    # Only plot if plot_entire_chain is True
    if plot_entire_chain:
        fig, ax = plt.subplots(1, 1, figsize=plot_size)

        print(f" --> Plotting: {v} in category {cat}")
        for proc in procs.keys():
            label, color = procs[proc]

            cat_mask = dfs[proc]['category'] == cat

            x = np.array(dfs[proc][v_dfs][cat_mask])

            # Event weight
            w = np.array(dfs[proc]['plot_weight'])[cat_mask]
            

            counts, bin_edges, _ = ax.hist(x, nbins, xrange, density = plot_fraction, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

            if proc == "background":
                # Plot the fitted exponential decay curve
                x_fit = np.linspace(xrange[0], xrange[1], 1000)
                y_fit = exponential_decay(x_fit, A, lambd)
                ax.plot(x_fit, y_fit, color="red", linestyle="--",
                        label=f"Exponential Fit\n$A={A:.2f}$, $\\lambda={lambd:.4f}$")

        ax.set_xlabel(sanitized_var_name)
        ax.set_ylabel("Events")

        if is_log_scale:
            ax.set_yscale("log")

        ax.legend(loc='best', ncol=2)

        hep.cms.label(f"category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

        plt.tight_layout()
        ext = f"_cat_{cat}"
        fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
        plt.show()
    
    

#breakpoint()#%%

#%%

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Confusion Matrix

    
# Define bins and labels for pt categories
bins = [0, 60, 120, 200, 300, np.inf]
labels = ['0-60', '60-120', '120-200', '200-300', '>300']

# Categorise 'HTXS_Higgs_pt_sel' into the same pt categories
for proc in procs.keys():
    if proc == "Data":
        continue  

    dfs[proc]['truth_category'] = pd.cut(dfs[proc]['HTXS_Higgs_pt_sel'], bins=bins, labels=labels, right=False)

confusion_matrices = generate_confusion_matrices(dfs, ["ttH", "ggH"], labels, normalised=True, plot=plot_entire_chain)
    
#%%

if plot_entire_chain:
    unfiltered_ttH_confusion_matrix(bins, labels, normalised=True) # To plots Unfiltered ttH to compare


#%%

from NLL import *

# Define signal window parameters
hists = {}

mass_bins = 5
v = 'mass'
v_dfs = v + "_sel"

# Initialize histogram data for each reconstructed category and process
for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        # Apply mask to categorize events by reconstructed category
        if proc == "background":
            hists[cat][proc] = np.array(background_estimates[cat][proc])
        else:
            cat_mask = dfs[proc]['category'] == cat
            hists[cat][proc] = np.histogram(
                dfs[proc][cat_mask][v_dfs], 
                mass_bins, 
                mass_range, 
                weights=dfs[proc][cat_mask]['true_weight']
            )[0]

ordered_hists = {key: hists[key] for key in labels}

conf_matrix = confusion_matrices['ttH']

combined_histogram = build_combined_histogram(ordered_hists, conf_matrix, signal='ttH')

mu_values = np.linspace(-100, 100, 1000)  # Range for scanning a single mu
mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]
bounds = [(0, 3) for _ in range(4)]  # Bounds for the other mu parameters

if plot_entire_chain:
    frozen_optimised_mus, profile_optimised_mus = perform_NLL_scan_and_profile_and_other_mus(mu_values, mus_initial, bounds, combined_histogram, signal='ttH', plot=True)


#%%

optimized_mus = [1,1,1,1,1]

plot_combined_histogram(combined_histogram, labels, processes_to_exclude = "background")
    
#%%

labels = ['0-60', '60-120', '120-200', '200-300', '>300'] # define labels again for hists

NLL_hessian_matrix = calc_Hessian_NLL(combined_histogram, optimized_mus)
print("Hessian Matrix:\n", np.array2string(NLL_hessian_matrix, precision=4, separator=' ', suppress_small=True))

try:
    covariant_matrix = np.linalg.inv(NLL_hessian_matrix)
    print("\nCovariant Matrix:\n", np.array2string(covariant_matrix, precision=4, separator=' ', suppress_small=True))
except np.linalg.LinAlgError:
    print("Error: Hessian matrix is singular and cannot be inverted. Check if the Hessian is non-singular.")

# Extract uncertainties for each mu parameter
uncertainties = np.sqrt(np.diag(covariant_matrix))
print("\nUncertainties in mu parameters:\n", np.array2string(uncertainties, precision=4, separator=', ', suppress_small=True))


correlation_matrix = covariance_to_correlation(covariant_matrix)
print("\nCorrelation Matrix:\n", np.array2string(correlation_matrix, precision=4, separator=' ', suppress_small=True))



if plot_entire_chain:
    plot_matrix(covariant_matrix, title = "Covrariant Matrix")

#%%

from Chi_Squared import *

quadratic_order = True


chi_squared_scans(optimized_mus, NLL_hessian_matrix, np.linspace(-2, 2, 1000), quadratic_order)

#%%

compare_chi_squared_scans(optimized_mus, NLL_hessian_matrix, np.linspace(-1, 1, 1000), plot_individuals = False)



#%%
def chi_squared_wrapper(cg_ctg, optimized_mus, NLL_hessian_matrix, quadratic_order):
    cg, ctg = cg_ctg  # Extract cg and ctg from the array
    return chi_squared_func(cg, ctg, optimized_mus, NLL_hessian_matrix, quadratic_order)


# Initial guess for c_g and c_tg
initial_guess = [0.0, 0.0]  # Adjust as needed

# Minimize the chi-squared function using the wrapper
result_chi2 = minimize(chi_squared_wrapper, initial_guess, args=(optimized_mus, NLL_hessian_matrix, quadratic_order), method='L-BFGS-B')

# Extract results
optimal_cg, optimal_ctg = result_chi2.x
min_chi_squared = result_chi2.fun

print(f"Optimal c_g: {optimal_cg}")
print(f"Optimal c_tg: {optimal_ctg}")
print(f"Minimum chi-squared: {min_chi_squared}")


#%%

chi_squared_grid(optimized_mus, NLL_hessian_matrix, np.linspace(-1, 1, 100), result_chi2.x, quadratic_order)


#%%
quadratic_order = True



chi2_hessian = compute_chi2_hessian(0, 0, optimized_mus, NLL_hessian_matrix, quadratic_order, epsilon=0.05)

print("Chi Squared Hessian Matrix:\n", np.array2string(chi2_hessian, precision=4, separator=' ', suppress_small=True))

try:
    chi2_covariant_matrix = np.linalg.inv(chi2_hessian)
    print("\nChi Squared Covariant Matrix:\n", np.array2string(chi2_covariant_matrix, precision=4, separator=' ', suppress_small=True))
except np.linalg.LinAlgError:
    print("Error: Hessian matrix is singular and cannot be inverted. Check if the Hessian is non-singular.")

if plot_entire_chain:
    plot_matrix(chi2_covariant_matrix,"$c_{tg}$","$c_g$", title = "$\chi^2$ Covariant Matrix")

# Extract uncertainties for each mu parameter
chi2_uncertainties = np.sqrt(np.diag(chi2_covariant_matrix))
print("\nChi Squared Uncertainties in parameters:\n", np.array2string(chi2_uncertainties, precision=4, separator=', ', suppress_small=True))


chi2_correlation_matrix = covariance_to_correlation(chi2_covariant_matrix)
print("\nChi Squared Correlation Matrix:\n", np.array2string(chi2_correlation_matrix, precision=4, separator=' ', suppress_small=True))

if plot_entire_chain:
    plot_matrix(chi2_correlation_matrix,"$c_{tg}$","$c_g$", title = "$\chi^2$ Correlation Matrix")
                                                                                                                                                                                                                          


