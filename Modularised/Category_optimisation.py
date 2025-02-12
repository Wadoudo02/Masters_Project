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
checkpoint = torch.load("neural_network.pth")

# Instantiate the model
loaded_model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
loaded_model.eval()


# Constants
total_lumi = 7.9804
target_lumi = 300

def bounds_of_wilson_coefficients(category_bounds):
    
    category_boundaries = sorted(category_bounds)
    
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
    
        # Remove rows with negative plot_weight from DataFrame
        dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]
    
        # Reweight to target lumi
        dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)
    
        # Calculate true weight: remove x10 multiplier for signal
        if proc in ['ggH', 'VBF', 'VH', 'ttH']:
            dfs[proc]['true_weight'] = dfs[proc]['plot_weight']/10
        else:
            dfs[proc]['true_weight'] = dfs[proc]['plot_weight']
            
        # Re-normalise the weights
        dfs[proc]["true_weight"] /= dfs[proc]["true_weight"].sum()
        
        dfs[proc]["true_weight"] *= 1000
    
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
        'frozen_cg_vals',
        'profile_cg_vals',
        'frozen_ctg_vals',
        'profile_ctg_vals'
    ]
    
    # Flatten only the values from these chosen keys
    flattened_list = []
    for key in keys_of_interest:
        if key in NLL_Results:
            flattened_list.extend(NLL_Results[key])


    
    return 10000 * np.sum([abs(num) for num in flattened_list])

#%%


print(bounds_of_wilson_coefficients([0, 0.24066145, 0.29167122, 0.33349041, 1]))
print(bounds_of_wilson_coefficients([0, 0.3488496281206608, 0.5095711573958397, 0.6702926866710186, 1]))

#%%


def optimize_category_bounds(fix_ends=False):
    """
    Minimises the scalar returned by `bounds_of_wilson_coefficients` 
    with respect to the internal category boundaries.
    
    If fix_ends=True, the first boundary is 0.0 and the last boundary is 1.0,
    and only the internal boundaries are optimised.
    Otherwise, all boundaries in [0, 1] are free to vary subject to sorting constraints.
    """
    
    # Number of categories and bin edges:
    # We want 4 categories → 5 edges in total.
    num_categories = 4
    num_bounds = num_categories + 1  # e.g. 5

    # ----------------------------------------------------------------------
    # CASE 1: Fix the first boundary at 0 and the last boundary at 1
    # ----------------------------------------------------------------------
    if fix_ends:
        # Then we only optimise the internal boundaries (3 of them for 4 categories)
        def objective(internal_bounds):
            # Reconstruct full boundaries: 0, x[0], x[1], x[2], 1
            # Sort to ensure monotonicity before passing to your function
            sorted_bounds = [0.0] + sorted(internal_bounds) + [1.0]
            return bounds_of_wilson_coefficients(sorted_bounds)

        # Constraints (SLSQP 'ineq' => must be ≥ 0) to ensure x[0] < x[1] < x[2] strictly
        # as well as 0 ≤ x[0] and x[2] ≤ 1. Here we just ensure they stay in (0, 1) 
        # and remain strictly increasing. 
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[1] - x[0]},  # x[1] > x[0]
            {'type': 'ineq', 'fun': lambda x: x[2] - x[1]},  # x[2] > x[1]
            {'type': 'ineq', 'fun': lambda x: x[0]},         # x[0] ≥ 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[2]}       # x[2] ≤ 1
        ]

        # A sensible initial guess for three internal boundaries in [0,1]:
        x0 = [0.24066145, 0.29167122, 0.33349041]


        result = minimize(
                    objective, 
                    x0, 
                    method='SLSQP', 
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10, 'disp': True}
                )
        if not result.success:
            print("Minimisation failed:", result.message)

        final_sorted_bounds = [0.0] + sorted(result.x) + [1.0]
        print("Optimised category boundaries (fixed ends):", final_sorted_bounds)
        print("Minimum objective value:", result.fun)
        return final_sorted_bounds, result.fun

    # ----------------------------------------------------------------------
    # CASE 2: Allow all boundaries (including the first & last) to float in [0,1]
    # ----------------------------------------------------------------------
    else:
        # We have 5 boundaries in total to find, x = [x0, x1, x2, x3, x4] 
        # which must satisfy 0 ≤ x0 < x1 < x2 < x3 < x4 ≤ 1
        def objective(free_bounds):
            sorted_bounds = sorted(free_bounds)
            return bounds_of_wilson_coefficients(sorted_bounds)

        constraints = []
        # We want x0 < x1, x1 < x2, etc. so:
        for i in range(num_bounds - 1):
            constraints.append(
                {'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]}  # x[i+1] > x[i]
            )
        # Also enforce x0 ≥ 0, x4 ≤ 1:
        constraints.append(
            {'type': 'ineq', 'fun': lambda x: x[0]}      # x[0] ≥ 0
        )
        constraints.append(
            {'type': 'ineq', 'fun': lambda x: 1 - x[-1]} # x[-1] ≤ 1
        )

        # An initial guess spaced evenly:
        x0 = [0.2, 0.24066145, 0.29167122, 0.33349041, 0.6]

        result = minimize(
            objective, x0, 
            method='SLSQP', 
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 300, 'disp': True}
        )
        if not result.success:
            print("Minimisation failed:", result.message)

        final_sorted_bounds = sorted(result.x)
        print("Optimised category boundaries (all free):", final_sorted_bounds)
        print("Minimum objective value:", result.fun)
        return final_sorted_bounds, result.fun
#%%
# Example usage

#best_bounds_fixed, min_obj_fixed = optimize_category_bounds(fix_ends=True)
#print("Fixed boundary result:", best_bounds_fixed, "Objective:", min_obj_fixed)

best_bounds_free, min_obj_free = optimize_category_bounds(fix_ends=False)
print("Free boundary result:", best_bounds_free, "Objective:", min_obj_free)

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
            [0.2, 0.24, 0.29, 0.33, 0.6],  # Original guess
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
best_bounds_free, min_obj_free = optimize_category_bounds_improved(fix_ends=True)
print("Optimized boundaries:", best_bounds_free)
print("Objective value:", min_obj_free)





