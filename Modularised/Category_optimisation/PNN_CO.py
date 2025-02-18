#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:21:55 2025

@author: wadoudcharbak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:16:18 2025

@author: wadoudcharbak
"""

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

import torch


from NN_utils import *

from scipy.optimize import minimize


# Load the model checkpoint
checkpoint = torch.load("data/neural_network_parameterised.pth")

# Instantiate the model
loaded_model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
loaded_model.eval()


# Constants
total_lumi = 7.9804
target_lumi = 300


#breakpoint()
# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    #"ttH_SMEFT" : ["ttH_SMEFT x 10", "green"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF x 10", "red"],
    "VH" : ["VH x 10", "orange"],
    #"Data" : ["Data", "green"]
}

plot_size = (12, 8)


cg = 0.3
ctg = 0.69


# Load dataframes


# Labels for the categories
labels = ['NN Cat A', 'NN Cat B', 'NN Cat C', 'NN Cat D'] # Labels for each category


dfs = {}
for i, proc in enumerate(procs.keys()):
    #print(f" --> Loading process: {proc}")
    
    if proc == "ttH_SMEFT":
        dfs[proc] = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
    else:
        dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]
   
    yield_weight = dfs[proc]["plot_weight"].sum()
   
    # Remove rows with negative plot_weight from DataFrame
    dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]
    
    dfs[proc]["plot_weight"] /= dfs[proc]["plot_weight"].sum()
    dfs[proc]["plot_weight"] *= yield_weight
   
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
    
    
    mask = dfs[proc]['n_jets_sel'] >= 0
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.4)
    mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    if proc == "ttH_SMEFT":
        dfs[proc] = add_SMEFT_weights(dfs[proc], cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

#%%


import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from numba import njit

@njit
def exponential_decay(x, A, lambd):
    """
    Returns A * exp(-lambd * (x - 120)).
    JIT-compiled for performance.
    """
    return A * np.exp(-lambd * (x - 120))


@njit
def exponential_decay_integral(A, lambd, x1, x2):
    """
    Analytical integral of A * exp[-lambd * (x - 120)] from x1 to x2:
    => -(A / lambd) * [exp(-lambd*(x - 120))]_x1^x2
    """
    return -(A / lambd) * (
        np.exp(-lambd*(x2 - 120)) - np.exp(-lambd*(x1 - 120))
    )


def background_fit_exponential(bin_centers, counts):
    """
    Perform a simple exponential fit using curve_fit.
    Ignores bins with zero counts.
    (Kept outside @njit because SciPy's curve_fit is not numba-compatible.)
    """
    non_zero_indices = counts > 0
    if np.sum(non_zero_indices) < 2:
        # Not enough non-zero bins to fit meaningfully
        return (0.0, 0.0)
    
    # Wrap exponential_decay with the needed signature for curve_fit
    popt, _ = curve_fit(
        lambda x, A, L: exponential_decay(x, A, L),
        bin_centers[non_zero_indices],
        counts[non_zero_indices]
    )
    return popt  # (A, lambd)


def get_bin_estimates(A, lambd, mass_range, mass_bins):
    """
    Compute the binned integral of A * exp(-lambd * (x-120)) in
    the range [mass_range[0], mass_range[1]] with mass_bins bins.
    """
    BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
    bin_estimates = []
    for i in range(len(BG_estimate_bin_edges) - 1):
        x1 = BG_estimate_bin_edges[i]
        x2 = BG_estimate_bin_edges[i + 1]
        integral_val = exponential_decay_integral(A, lambd, x1, x2)
        bin_estimates.append(integral_val)
    return bin_estimates


def bounds_of_wilson_coefficients(category_bounds):
    """
    A faster version of 'bounds_of_wilson_coefficients' that:
      - avoids quad by using an analytical integral,
      - minimises repeated Pandas indexing,
      - uses Numba-accelerated exponential functions,
      - otherwise preserves logic close to the original.
    """
    category_boundaries = sorted(category_bounds)
    cats_unique = labels.copy()

    # ~~~~~ 1) Compute NN probabilities and categories in one pass
     # Extract the features for NN input
    features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
    features = [f"{feature}_sel" for feature in features]
    features.append("cg")
    features.append("ctg")
    
    for proc in procs.keys():

        N = len(dfs[proc])
        
        dfs[proc]["cg"]  = np.random.uniform(low=cg_min,  high=cg_max,  size=N)
        dfs[proc]["ctg"] = np.random.uniform(low=ctg_min, high=ctg_max, size=N)
        
        if not all(feature in dfs[proc].columns for feature in features):
            raise ValueError(f"Missing one or more required features in process {proc}")
    
        # Prepare the input tensor for the NN
        nn_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
        # Get NN predictions
        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()


        # Store NN probabilities + categories back into df
        dfs[proc]["NN_probabilities"] = probabilities
        dfs[proc]["category"] = pd.cut(
            probabilities,
            bins=category_boundaries,
            labels=labels,
            include_lowest=True
        )

    # ~~~~~ 2) Background estimates via exponential fit
    background_estimates = {}
    mass_range = (120, 130)
    mass_bins = 5

    v = "mass"
    v_dfs = v + "_sel"

    for cat in cats_unique:
        nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]

        for proc in procs.keys():
            df_proc = dfs[proc]
            cat_mask = (df_proc["category"] == cat).values  # Faster boolean array

            xvals = df_proc[v_dfs].values[cat_mask]
            wvals = df_proc["plot_weight"].values[cat_mask]

            # Build histogram
            counts, bin_edges = np.histogram(xvals, bins=nbins, range=xrange, weights=wvals)

            if proc == "background":
                # Fit an exponential to these bin_counts
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                A, lambd = background_fit_exponential(bin_centers, counts)

                # Use analytical integral to get binned BG estimate in [120, 130]
                bin_estimates = get_bin_estimates(A, lambd, mass_range, mass_bins)

                # Store
                if cat not in background_estimates:
                    background_estimates[cat] = {}
                background_estimates[cat][proc] = bin_estimates

    # ~~~~~ 3) Combine histogram (use your existing function)
    combined_hist, hists_by_cat = build_combined_histogram_NN(
        dfs, procs, cats_unique, background_estimates,
        mass_var="mass_sel", weight_var="true_weight",
        mass_range=mass_range, mass_bins=mass_bins
    )

    # ~~~~~ 4) Category-wise averages
    params = ["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]
    cat_averages = {}

    for cat in cats_unique:
        df_cat = dfs["ttH"][dfs["ttH"]["category"] == cat]
        cat_averages[cat] = {}
        for param in params:
            cat_averages[cat][param] = get_weighted_average(df_cat, param, "true_weight")

    # ~~~~~ 5) Build final histograms in [120, 130] for each process/category
    hists = {}
    for cat in cats_unique:
        hists[cat] = {}
        for proc in procs.keys():
            if proc == "background":
                # Directly use the exponential-fit estimates
                hists[cat][proc] = np.array(background_estimates[cat][proc])
            else:
                # Build a histogram from real data
                df_proc = dfs[proc]
                cat_mask = (df_proc["category"] == cat).values
                xvals = df_proc[v_dfs].values[cat_mask]
                wvals = df_proc["true_weight"].values[cat_mask]

                hists[cat][proc], _ = np.histogram(
                    xvals, mass_bins, mass_range, weights=wvals
                )

    # ~~~~~ 6) NLL scans
    quadratic_order = True
    scan_points = np.linspace(-2, 2, 100)
    NLL_Results = NN_NLL_scans(hists, scan_points, cat_averages,
                               quadratic_order, plot=False)

    # Choose keys of interest to flatten
    keys_of_interest = ["profile_cg_vals", "profile_ctg_vals"]
    flattened_list = []
    for key in keys_of_interest:
        if key in NLL_Results:
            # Hypothetically these are arrays or lists
            flattened_list.append(NLL_Results[key][1])
            flattened_list.append(NLL_Results[key][2])

    # Final result
    return 10000 * np.sum([abs(num) for num in flattened_list])

#print(bounds_of_wilson_coefficients([0, 0.24066145, 0.29167122, 0.33349041, 1]))
#print(bounds_of_wilson_coefficients([0, 0.3488496281206608, 0.5095711573958397, 0.6702926866710186, 1]))



#%%


'''
boundary_values = np.linspace(0.21, 0.5, 20)  # 20 points from 0.6 to 0.9
results = []

for b2 in boundary_values:
    # category_bounds = [0, 0.2, 0.5, b3, 1]
    val = bounds_of_wilson_coefficients([0, 0.2, b2, 0.6, 1])
    results.append(val)

plt.figure(figsize=(7,5))
plt.plot(boundary_values, results, marker='o')
plt.xlabel("Fourth boundary")
plt.ylabel("Objective value")
plt.title("1D scan of objective vs. one NN boundary")
plt.grid(True)
plt.show()

# For 4 categories, we have 5 boundaries: [0, b1, b2, 1].
# (Here we assume we only want 3 categories for illustration.
#  If you truly need 4 categories, you’ll want [0, b1, b2, b3, 1]
#  in a triple-nested loop. The principle is the same.)

#%%

n_points = 20
b1_range = np.linspace(0.2, 0.6, n_points)
b2_range = np.linspace(0.2, 0.6, n_points)

value_grid = np.zeros((n_points, n_points))

for i, b1 in enumerate(b1_range):
    for j, b2 in enumerate(b2_range):
        # Make sure b2 > b1 so the bins make sense
        if b2 > b1:
            category_bounds = [0, b1, b2, 0.7, 1]  
            obj_val = bounds_of_wilson_coefficients(category_bounds)
            value_grid[i, j] = obj_val
        else:
            # Invalid region or force it to be NaN
            value_grid[i, j] = np.nan

# Now produce a contour or heatmap plot
B1, B2 = np.meshgrid(b2_range, b1_range)  # watch ordering

plt.figure(figsize=(8,6))
contour = plt.contourf(B1, B2, value_grid, levels=30, cmap='viridis')
plt.colorbar(contour, label='Objective Value')
plt.xlabel("Boundary b2")
plt.ylabel("Boundary b1")
plt.title("2D scan of objective vs. two NN boundaries")
plt.show()

#%%
'''

#%%

import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    """
    Wrapper for bounds_of_wilson_coefficients that handles the fixed boundaries
    and ensures ordering of intermediate points.
    
    Args:
        x: Array of 3 values representing the intermediate boundaries
    
    Returns:
        float: The objective function value
    """
    # Sort the intermediate boundaries to maintain order
    x_sorted = np.sort(x)
    
    # Create full boundary array with fixed endpoints
    full_boundaries = np.array([0.0] + list(x_sorted) + [1.0])
    
    # Check if boundaries are too close together
    if np.min(np.diff(full_boundaries)) < 0.05:  # Minimum gap of 0.05
        return 1e10  # Return large value if boundaries are too close
    
    try:
        return bounds_of_wilson_coefficients(full_boundaries)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 1e10  # Return large value if evaluation fails

# Initial guess for the intermediate boundaries
initial_guess = np.array([0.25, 0.5, 0.70])

# Define bounds for the optimization
bounds = [(0.05, 0.95) for _ in range(3)]  # Each boundary must be between 0.05 and 0.95

# Run the optimization
result = minimize(
    objective_function,
    initial_guess,
    method='Nelder-Mead',
    bounds=bounds,
    options={
        'maxiter': 1000,
        'xatol': 1e-4,
        'fatol': 1e-4
    }
)

# Get the optimal boundaries
optimal_intermediate_boundaries = np.sort(result.x)
optimal_full_boundaries = np.array([0.0] + list(optimal_intermediate_boundaries) + [1.0])

print("Optimization completed:")
print(f"Success: {result.success}")
print(f"Number of iterations: {result.nit}")
print(f"Final objective value: {result.fun}")
print("\nOptimal boundaries:")
print(optimal_full_boundaries)