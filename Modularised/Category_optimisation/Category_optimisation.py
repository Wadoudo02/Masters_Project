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
checkpoint = torch.load("data/neural_network.pth")

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


def bounds_of_wilson_coefficients(category_bounds):
    
    category_boundaries = sorted(category_bounds)
    
    for i, proc in enumerate(procs.keys()):
        # Extract the features for NN input
        features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
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

    
    
    
    cats_unique = labels.copy()
    
    def exponential_decay(x, A, lambd):
        return A * np.exp(-lambd * (x - 120))
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot diphoton mass distribution in each category
    
    background_estimates = {}
    
    mass_range = (120, 130)  # Needed here to get BG estimate
    mass_bins = 5
    
    v = "mass"
    v_dfs = v + "_sel"
    for cat in cats_unique:
        #print(f" --> Processing: {v} in category {cat}")
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
    
                #print(f"Background estimates for category {cat}: {bin_estimates}")
    
                # Store the result
                if cat not in background_estimates:
                    background_estimates[cat] = {}
                background_estimates[cat][proc] = bin_estimates
    
         
    
    
    combined_hist, hists_by_cat = build_combined_histogram_NN(dfs, procs, cats_unique, background_estimates, mass_var="mass_sel", 
                             weight_var="true_weight", mass_range=(120, 130), mass_bins=5)
    
    
    
    # Parameters for which we want category-wise averages
    params = ["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]
    
    cat_averages = {}
    
    
    for cat in cats_unique:
        # Slice df for this category
        df_cat = dfs["ttH"][dfs["ttH"]["category"] == cat]
        
        # Store each parameter's weighted mean
        cat_averages[cat] = {}
        for param in params:
            cat_averages[cat][param] = get_weighted_average(df_cat, param, "true_weight")
    
    
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
                
    
    
    quadratic_order = True
    
    
    
    NLL_Results = NN_NLL_scans(hists, np.linspace(-1, 1, 1000), cat_averages, quadratic_order, plot=False)
    
    # Specify only the keys of interest
    keys_of_interest = [
        'profile_cg_vals',
        'profile_ctg_vals'
    ]
    
    # Flatten only the values from these chosen keys
    flattened_list = []
    for key in keys_of_interest:
        if key in NLL_Results:
            #flattened_list.extend(NLL_Results[key])
            flattened_list.append(NLL_Results[key][1])
            flattened_list.append(NLL_Results[key][2])
    
    #breakpoint()
    
    return 10000 * np.sum([abs(num) for num in flattened_list])



#print(bounds_of_wilson_coefficients([0, 0.24066145, 0.29167122, 0.33349041, 1]))
#print(bounds_of_wilson_coefficients([0, 0.3488496281206608, 0.5095711573958397, 0.6702926866710186, 1]))



#%%

def optimize_category_bounds_improved(fix_ends=False):
    """
    Improved optimization of category boundaries using multiple strategies
    """
    num_categories = 4
    num_bounds = num_categories + 1

    if fix_ends:
        def objective(internal_bounds):
            sorted_bounds = [0.0] + sorted(internal_bounds) + [1.0]
            return bounds_of_wilson_coefficients(sorted_bounds)

        bounds = [(0.01, 0.99) for _ in range(num_bounds-2)]
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.01},  # Minimum gap between boundaries
            {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 0.01},
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[2]}
        ]

        # Try multiple initial points
        best_result = None
        best_value = float('inf')
        
        initial_guesses = [
            [0.24, 0.29, 0.33],  # Your original guess
            [0.2, 0.4, 0.6],     # Evenly spread
            [0.3, 0.4, 0.5],     # Center-focused
            [0.1, 0.3, 0.5]      # Lower-focused
        ]

        for x0 in initial_guesses:
            # Try with different optimizers
            try:
                # SLSQP with tighter tolerances
                result_slsqp = minimize(
                    objective, 
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-12,
                        'eps': 1e-8,
                        'disp': True
                    }
                )
                
                if result_slsqp.success and result_slsqp.fun < best_value:
                    best_result = result_slsqp
                    best_value = result_slsqp.fun

                # Try Nelder-Mead as well (without constraints)
                result_nm = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-13
                    }
                )
                
                if result_nm.fun < best_value:
                    best_result = result_nm
                    best_value = result_nm.fun

            except Exception as e:
                print(f"Optimization failed for x0={x0}: {str(e)}")
                continue

        if best_result is None:
            raise ValueError("All optimization attempts failed")

        final_sorted_bounds = [0.0] + sorted(best_result.x) + [1.0]
        return final_sorted_bounds, best_result.fun

    else:
        def objective(bounds):
            return bounds_of_wilson_coefficients(sorted(bounds))

        bounds = [(0.01, 0.99) for _ in range(num_bounds)]
        
        # Create constraints for minimum spacing between boundaries
        constraints = []
        min_gap = 0.01  # Minimum gap between consecutive boundaries
        
        for i in range(num_bounds - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: sorted(x)[i+1] - sorted(x)[i] - min_gap
            })

        # Try multiple initial points with different optimization strategies
        best_result = None
        best_value = float('inf')
        
        initial_guesses = [
            #[0.0986808922572032, 0.31811643626238006, 0.5, 0.6818835637376199, 0.9013191077427969],  # Original guess
            np.linspace(0.1, 0.9, num_bounds),  # Linear spread
            np.array([0.1, 0.2, 0.5, 0.7, 0.9]),  # Asymmetric spread
            np.array([0.2, 0.3, 0.4, 0.5, 0.6])   # Center-focused
        ]

        for x0 in initial_guesses:
            try:
                # Try SLSQP with tighter tolerances
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-12,
                        'eps': 1e-8,
                        'disp': True
                    }
                )
                
                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun

                # Try trust-constr as an alternative
                result_trust = minimize(
                    objective,
                    x0,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'xtol': 1e-8,
                        'gtol': 1e-8
                    }
                )
                
                if result_trust.fun < best_value:
                    best_result = result_trust
                    best_value = result_trust.fun

            except Exception as e:
                print(f"Optimization failed for x0={x0}: {str(e)}")
                continue

        if best_result is None:
            raise ValueError("All optimization attempts failed")

        final_sorted_bounds = sorted(best_result.x)
        return final_sorted_bounds, best_result.fun

# Example usage:
best_bounds_free, min_obj_free = optimize_category_bounds_improved(fix_ends=False)
print("Optimized boundaries:", best_bounds_free)
print("Objective value:", min_obj_free)

#%%



boundary_values = np.linspace(0.21, 0.6, 20)  # 20 points from 0.6 to 0.9
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
b1_range = np.linspace(0.0, 0.8, n_points)
b2_range = np.linspace(0.2, 1.0, n_points)

value_grid = np.zeros((n_points, n_points))

for i, b1 in enumerate(b1_range):
    for j, b2 in enumerate(b2_range):
        # Make sure b2 > b1 so the bins make sense
        if b2 > b1:
            category_bounds = [0, b1, b2, 1]  
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
def objective(mid_bounds):
    """
    mid_bounds is [x1, x2, x3], with 0 < x1 < x2 < x3 < 1.
    We build [0, x1, x2, x3, 1] and evaluate the function.
    """
    category_bounds = [0.0] + sorted(mid_bounds) + [1.0]
    return bounds_of_wilson_coefficients(category_bounds)

min_prob = float("-inf")
max_prob = float("inf")

lower = min_prob+0.1
upper = max_prob-0.1
init_guess = [0.2, 0.5, 0.7]
#init_guess = [0.4905974388217408, 0.5436479211218064, 0.6436479210861633, 0.7436479061841965]

bounds = [(lower, upper), (lower, upper), (lower, upper)]

min_sep = 0.05
options = {'eps': 1e-4, 'ftol': 1e-6}  # Reduce step size and tolerance
# Define constraints to ensure increasing order
constraints = [{'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - min_sep} for i in range(len(init_guess) - 1)]
#constraints = [{'type': 'ineq', 'fun': lambda x: x[i+1] - x[i]} for i in range(len(init_guess) - 1)]


res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=options)

# Print the result
print("Optimization result:")
print(res)

# Extract the optimized values
optimized_values = res.x
print("Optimized values:", optimized_values)
'''
