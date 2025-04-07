#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:11:28 2025

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
checkpoint = torch.load("data/neural_network_new_func.pth")

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



dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    
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
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    if proc == "ttH_SMEFT":
        dfs[proc] = add_SMEFT_weights(dfs[proc], cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

import json

# Load the probability values
with open("data/proba_values_new_func.json", "r") as json_file:
    proba_data = json.load(json_file)

max_proba = proba_data["max_proba"]
min_proba = proba_data["min_proba"]

print(f"Max Probability: {max_proba}, Min Probability: {min_proba}")


#%%


import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from numba import njit

def create_category_boundaries(
    old_boundaries=None, 
    no_of_categories=4
):
    """
    Given old boundaries (length 5, for 4 original categories) at quantiles 0,0.25,0.5,0.75,1,
    produce new boundaries (length no_of_categories+1) preserving the overall 'shape'.
    
    Args:
        old_boundaries (list or numpy array): 
            The known boundaries of the 'old' distribution. E.g. [0., 0.35235969, 
            0.51631691, 0.71428785, 1.]
        no_of_categories (int): 
            Number of categories desired in the new distribution.
            
    Returns:
        numpy array of new boundaries (length no_of_categories + 1).
    """
    # If no old boundaries are provided, use your defaults
    if old_boundaries is None:
        old_boundaries = np.array([0., 0.35235969, 0.51631691, 0.71428785, 1.])
    else:
        old_boundaries = np.array(old_boundaries)

    # These are the quantiles at which old_boundaries are defined
    #   0 -> old_boundaries[0], 
    #   0.25 -> old_boundaries[1], etc.
    old_q = np.linspace(0, 1, len(old_boundaries))  # e.g. [0., 0.25, 0.5, 0.75, 1.]

    # We want (no_of_categories+1) new boundaries
    # Subdivide 0..1 into no_of_categories equal quantile steps
    new_q = np.linspace(0, 1, no_of_categories + 1)

    # Interpolate the old boundaries at these new quantile steps
    new_boundaries = np.interp(new_q, old_q, old_boundaries)

    # Force first boundary=0 and last boundary=1, just in case
    new_boundaries[0]  = 0.0
    new_boundaries[-1] = 1.0

    return new_boundaries

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


def positive_bounds_asafunctionof_cat_numnber(no_of_categories):
    """
    A faster version of 'bounds_of_wilson_coefficients' that:
      - avoids quad by using an analytical integral,
      - minimises repeated Pandas indexing,
      - uses Numba-accelerated exponential functions,
      - otherwise preserves logic close to the original.
    """
    print("Number of Categories = ", no_of_categories)
    
    
    if no_of_categories <= 4:
        # Set exact edges to 0 and 1 for consistency
        category_boundaries = create_category_boundaries( 
            no_of_categories=no_of_categories)
    else:
        # Calculate the range
        proba_range = max_proba - min_proba
        
        # Generate boundaries
        category_boundaries = [
            min_proba + i * (proba_range / no_of_categories)
            for i in range(no_of_categories + 1)
        ]
    
    # Set exact edges to 0 and 1 for consistency
    category_boundaries[0] = 0
    category_boundaries[-1] = 1
    
    
    labels = labels = [f"cat {i + 1}" for i in range(no_of_categories)]
    
    cats_unique = labels.copy()

    # ~~~~~ 1) Compute NN probabilities and categories in one pass
     # Extract the features for NN input
    features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
    features = [f"{feature}_sel" for feature in features]
    features.append("cg")
    features.append("ctg")
    
    for proc in procs.keys():

         # Extract the features for NN input
        features = ["deltaR", "HT", "n_jets", "delta_phi_gg", "pt"]
        features = [f"{feature}_sel" for feature in features]

        if not all(feature in dfs[proc].columns for feature in features):
            raise ValueError(f"Missing one or more required features in process {proc}")
    
        # Prepare the input tensor for the NN
        nn_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
        # Get NN predictions
        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()
            
        # Add the probabilties as a category
        dfs[proc]["NN_probabilities"] = probabilities
    
        # Categorise based on probabilities
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

            df_proc = dfs[proc]
            cat_mask = (df_proc["category"] == cat).values
            xvals = df_proc[v_dfs].values[cat_mask]
            wvals = df_proc["true_weight"].values[cat_mask]

            hists[cat][proc], _ = np.histogram(
                xvals, mass_bins, mass_range, weights=wvals
            )

    # ~~~~~ 6) NLL scans
    quadratic_order = True
    #breakpoint()
    NLL_Results = NN_NLL_scans(hists, np.linspace(-10, 10, 1000), cat_averages, quadratic_order, mass_bins, plot = False)
    
    return NLL_Results['profile_cg_vals'][1], NLL_Results['profile_ctg_vals'][1], NLL_Results['profile_cg_vals'][2], NLL_Results['profile_ctg_vals'][2]

# Define the range of mass bin values
cat_no_values = list(range(1, 11))

# Store results
pos_cg_bounds = []
pos_ctg_bounds = []
neg_cg_bounds = []
neg_ctg_bounds = []

# Loop over each mass bin value and get the bounds
for cat_no in cat_no_values:
    positive_cg_bound, positive_ctg_bound, negitive_cg_bound, negitive_ctg_bound = positive_bounds_asafunctionof_cat_numnber(cat_no)
    pos_cg_bounds.append(positive_cg_bound)
    pos_ctg_bounds.append(positive_ctg_bound)
    neg_cg_bounds.append(negitive_cg_bound)
    neg_ctg_bounds.append(negitive_ctg_bound)

#%%

cat_no_values = list(range(1, 11))

# Plot for positive cg bound
plt.figure(figsize=(8, 5))
plt.plot(cat_no_values, pos_cg_bounds, marker='o')
plt.title("Positive $c_g$ Bound vs Number of Categories")
plt.xlabel("Number of Categories")
plt.ylabel("Positive $c_g$ Bound")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for positive ctg bound
plt.figure(figsize=(8, 5))
plt.plot(cat_no_values, pos_ctg_bounds, marker='o')
plt.title("Positive $c_{tg}$ Bound vs Number of Categories")
plt.xlabel("Number of Categories")
plt.ylabel("Positive $c_{tg}$ Bound")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for positive cg bound
plt.figure(figsize=(8, 5))
plt.plot(cat_no_values, neg_cg_bounds, marker='o')
plt.title("Negitive $c_g$ Bound vs Number of Categories")
plt.xlabel("Number of Categories")
plt.ylabel("Negitive $c_g$ Bound")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for positive ctg bound
plt.figure(figsize=(8, 5))
plt.plot(cat_no_values, neg_ctg_bounds, marker='o')
plt.title("Negitive $c_{tg}$ Bound vs Number of Categories")
plt.xlabel("Number of Categories")
plt.ylabel("Negitive $c_{tg}$ Bound")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

np.save("data/NN_pos_cg_bounds.npy", pos_cg_bounds)
np.save("data/NN_pos_ctg_bounds.npy", pos_ctg_bounds)

