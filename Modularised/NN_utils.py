#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:56:02 2025

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


from NLL import *
from Chi_Squared import *

# Define the NeuralNetwork class as in the original script
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)


# SMEFT weighting function
def add_SMEFT_weights(proc_data, cg, ctg, name="new_weights", quadratic=False):
    proc_data[name] = proc_data['plot_weight'] * (1 + proc_data['a_cg'] * cg + proc_data['a_ctgre'] * ctg)
    if quadratic:
        proc_data[name] += (
            (cg ** 2) * proc_data["b_cg_cg"]
            + cg * ctg * proc_data["b_cg_ctgre"]
            + (ctg ** 2) * proc_data["b_ctgre_ctgre"]
        )
    return proc_data

def build_combined_histogram_NN(
    dfs, procs, categories, background_estimates,
    mass_var="mass_sel", weight_var="plot_weight", 
    mass_range=(120, 130), mass_bins=5):
    """
    Builds a single histogram with (num_categories * mass_bins) bins,
    splitting mass into 'mass_bins' in [mass_range], for each of the NN categories.

    Parameters
    ----------
    dfs : dict of pd.DataFrame
        Dictionary of DataFrames, keyed by process.
    procs : dict
        Dictionary describing processes, e.g. { "background": ["Background", "black"], ... }.
        We only need the keys to iterate here.
    categories : list of str
        List of category labels, e.g. ['NN Cat A', 'NN Cat B', 'NN Cat C', 'NN Cat D'].
    background_estimates : dict
        A nested dictionary of the form:
            {
              'NN Cat A': {'background': [bin_0, bin_1, ..., bin_(mass_bins-1)]},
              'NN Cat B': {'background': [ ... ]},
              ...
            }
        giving the background yield estimates for each category & mass bin.
    mass_var : str, optional
        Name of the column in dfs[proc] with the diphoton mass (default "mass_sel").
    weight_var : str, optional
        Name of the column with the event weight (default "plot_weight").
    mass_range : tuple, optional
        (min_mass, max_mass). Default is (120, 130).
    mass_bins : int, optional
        Number of bins to slice mass_range into. Default is 5.

    Returns
    -------
    combined_histogram : dict
        A dictionary of 1D arrays keyed by process. Each array has
        (num_categories * mass_bins) bins, representing (category_idx × mass_bin_idx).
    hists_by_category : dict
        An intermediate dictionary that holds the (mass_bins) histogram
        for each category & process.
        e.g. hists_by_category['NN Cat A']['background'] = [ ... 5 bin yields ... ]
    """

    # 1. Build per-category, per-process histograms
    hists_by_category = {}
    for cat in categories:
        cat_dict = {}
        for proc in procs:
            if proc == "background":
                # Use the *fitted* background estimates (already integrated by mass bin)
                bin_counts = np.array(background_estimates[cat]["background"])
            else:
                # Normal histogram from the data frames
                cat_mask = (dfs[proc]["category"] == cat)
                mass_vals = dfs[proc][mass_var][cat_mask]
                weights   = dfs[proc][weight_var][cat_mask]

                bin_counts, _ = np.histogram(
                    mass_vals,
                    bins=mass_bins,
                    range=mass_range,
                    weights=weights
                )

            cat_dict[proc] = bin_counts
        hists_by_category[cat] = cat_dict

    # 2. Combine into a single histogram ( (num_categories * mass_bins) bins ) per process
    num_categories = len(categories)
    combined_bins = num_categories * mass_bins

    # Prepare the output structure
    combined_histogram = {}
    for proc in procs:
        combined_histogram[proc] = np.zeros(combined_bins)

    # Fill the combined histogram
    for i, cat in enumerate(categories):
        for proc in procs:
            bin_yields = hists_by_category[cat][proc]  # length == mass_bins
            for mbin in range(mass_bins):
                combined_bin_idx = i * mass_bins + mbin
                combined_histogram[proc][combined_bin_idx] += bin_yields[mbin]

    return combined_histogram, hists_by_category


def get_weighted_average(df, value_col, weight_col):
    """
    Computes the weighted average of 'value_col' in 'df',
    using 'weight_col' as the weighting column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    value_col : str
        The name of the column whose values you want to average.
    weight_col : str
        The name of the column representing the weights.

    Returns
    -------
    float
        The weighted average of 'value_col'.
        Returns NaN if the total weight is zero or if
        df is empty.
    """
    if df.empty:
        return np.nan

    values = df[value_col].to_numpy()
    weights = df[weight_col].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return np.nan

    return (values * weights).sum() / total_weight



def mu_c_NN(cg, ctg, cat_averages, quadratic=True):
    """
    Calculates mu (signal strength) for each category as a function of cg and ctg,
    based on the average SMEFT coefficients stored in 'cat_averages'.

    Parameters
    ----------
    cg : float
        Wilson coefficient c_g.
    ctg : float
        Wilson coefficient c_tg (ctgre in your notation).
    cat_averages : dict
        A nested dictionary of the form:
          {
            'NN Cat A': { 'a_cg': val, 'a_ctgre': val, 'b_cg_cg': val, ... },
            'NN Cat B': { ... },
            ...
          }
        giving the *weighted* average (a_cg, a_ctgre, b_cg_cg, etc.) for each category.
    quadratic : bool, optional
        Whether to include the quadratic terms in the SMEFT parameterisation.

    Returns
    -------
    mus : np.ndarray
        1D array of mu for each category, in the same order as cat_averages.keys().
    """
    cats = list(cat_averages.keys())  # e.g. ["NN Cat A", "NN Cat B", "NN Cat C", "NN Cat D"]
    mus = []
    for cat in cats:
        a_cg     = cat_averages[cat]["a_cg"]
        a_ctgre  = cat_averages[cat]["a_ctgre"]
        b_cg_cg  = cat_averages[cat]["b_cg_cg"]
        b_cg_ctg = cat_averages[cat]["b_cg_ctgre"]
        b_ctg_ctg = cat_averages[cat]["b_ctgre_ctgre"]

        # Start with 1 + linear terms
        mu_val = 1 + cg*a_cg + ctg*a_ctgre

        # Optionally add quadratic SMEFT pieces
        if quadratic:
            mu_val += (cg**2)*b_cg_cg + (cg*ctg)*b_cg_ctg + (ctg**2)*b_ctg_ctg

        mus.append(mu_val)
    
    return np.array(mus)


def calc_NLL_Simple(hists, mu_array, signal="ttH"):
    """
    Calculates the Poisson negative log-likelihood for multiple categories,
    with each category having 5 bins, ignoring additive n! constants.

    Parameters:
    -----------
    hists : dict
        A nested dict: { category_name: { process: np.array(5) } }
    mu_array : list or np.ndarray
        Array of signal strength modifiers, one per category.
        e.g. mu_array[0] is the mu for "NN Cat A", etc.
    signal : str
        Name of the process to be scaled by mu in each category.

    Returns:
    --------
    float
        The negative log-likelihood.
    """
    categories = list(hists.keys())  # e.g. ["NN Cat A", "NN Cat B", "NN Cat C", "NN Cat D"]
    nll = 0.0
    
    for i_cat, cat in enumerate(categories):
        # 5 bins in each category
        # hists[cat][proc] => array of length 5
        for b in range(5):
            # Observed yield
            n_b = 0.0
            # Expected yield
            e_b = 0.0
            for proc, yield_array in hists[cat].items():
                if proc == signal:
                    e_b += mu_array[i_cat] * yield_array[b]
                else:
                    e_b += yield_array[b]
                # For an Asimov dataset, the "observed" is also the sum of all processes
                n_b += yield_array[b]
            # Poisson NLL for this bin
            # log(e_b + tiny) to avoid log(0)
            nll += (e_b - n_b * np.log(e_b + 1e-10))

    return nll





def NN_NLL_scans(
    hists,
    range_of_values,
    cat_averages,
    quadratic_order=True,
    fixed_ctg=0,
    fixed_cg=0,
    plot = True
):
    cg_values = range_of_values.copy()
    ctg_values = range_of_values.copy()

    profile_NN_NLL_vals_cg = []
    profile_NN_NLL_vals_ctg = []
    minimized_ctg_for_cg = []  # Store minimized c_tg values for each c_g
    minimized_cg_for_ctg = []  # Store minimized c_g values for each c_tg

    order = "Quadratic" if quadratic_order else "First"

    # Profile scans for c_g
    for cg in cg_values:
        result = minimize(lambda ctg: calc_NLL_Simple(hists, mu_c_NN(cg, ctg, cat_averages = cat_averages, quadratic = quadratic_order )), x0=0)
        profile_NN_NLL_vals_cg.append(result.fun)
        minimized_ctg_for_cg.append(result.x[0])

    # Profile scans for c_tg
    for ctg in ctg_values:
        result = minimize(lambda cg: calc_NLL_Simple(hists, mu_c_NN(cg, ctg, cat_averages = cat_averages, quadratic = quadratic_order )), x0=0)
        profile_NN_NLL_vals_ctg.append(result.fun)
        minimized_cg_for_ctg.append(result.x[0])

    # Frozen scans    

    frozen_NN_NLL_vals_cg = [calc_NLL_Simple(hists, mu_c_NN(cg, fixed_ctg, cat_averages = cat_averages, quadratic = quadratic_order ))  for cg in cg_values]
    frozen_NN_NLL_vals_ctg = [calc_NLL_Simple(hists, mu_c_NN(fixed_cg, ctg, cat_averages = cat_averages, quadratic = quadratic_order )) for ctg in ctg_values]

    #breakpoint()

    # Convert NLL values to 2ΔNLL
    frozen_NN_NLL_vals_cg = TwoDeltaNLL(frozen_NN_NLL_vals_cg)
    frozen_NN_NLL_vals_ctg = TwoDeltaNLL(frozen_NN_NLL_vals_ctg)
    
    profile_NN_NLL_vals_cg = TwoDeltaNLL(profile_NN_NLL_vals_cg)
    profile_NN_NLL_vals_ctg = TwoDeltaNLL(profile_NN_NLL_vals_ctg)
    
    # Add Labels for cg
    
    frozen_cg_vals = find_confidence_interval(frozen_NN_NLL_vals_cg, cg_values, min(frozen_NN_NLL_vals_cg), 1)
    profile_cg_vals = find_confidence_interval(profile_NN_NLL_vals_cg, cg_values, min(profile_NN_NLL_vals_cg), 1)
    
    frozen_cg_label = add_val_label(frozen_cg_vals)
    profile_cg_label = add_val_label(profile_cg_vals)
    
    # Add Labels for ctg
    
    frozen_ctg_vals = find_confidence_interval(frozen_NN_NLL_vals_ctg, ctg_values, min(frozen_NN_NLL_vals_ctg), 1)
    profile_ctg_vals = find_confidence_interval(profile_NN_NLL_vals_ctg, ctg_values, min(profile_NN_NLL_vals_ctg), 1)
    
    frozen_ctg_label = add_val_label(frozen_ctg_vals)
    profile_ctg_label = add_val_label(profile_ctg_vals)

    if plot:    

        # Plot results
    
        plt.figure(figsize=(16, 12))
        plt.suptitle(f"Frozen and Profile NLL Scans ({order} Order)", fontsize=30)
    
        # c_g scan
        plt.subplot(2, 2, 1)
        plt.plot(cg_values, frozen_NN_NLL_vals_cg, label=f"Frozen $NLL(c_g, c_{{tg}} = {fixed_ctg})$ {frozen_cg_label}")
        plt.plot(cg_values, profile_NN_NLL_vals_cg, label=f"Profile $NLL(c_g$ min($c_{{tg}}$)) {profile_cg_label}")
        plt.axhline(1, color='red', linestyle='--', label="68% CL ($2\\Delta NLL = 1$)")
        plt.xlabel(r"Wilson coefficient $c_{g}$")
        plt.ylabel("2$\\Delta$NLL")
        plt.legend()
        plt.ylim(0, 10)
        plt.grid()
    
        # Minimized c_tg for c_g scan
        plt.subplot(2, 2, 2)
        plt.plot(cg_values, minimized_ctg_for_cg, label=r"Minimised $c_{tg}$ for each $c_{g}$")
        plt.xlabel(r"Wilson coefficient $c_{g}$")
        plt.ylabel(r"Minimised $c_{tg}$")
        plt.legend()
        plt.grid()
    
        # c_tg scan
        plt.subplot(2, 2, 3)
        plt.plot(ctg_values, frozen_NN_NLL_vals_ctg, label=f"Frozen $NLL(c_{{tg}}, c_g = {fixed_cg})$ {frozen_ctg_label}")
        plt.plot(ctg_values, profile_NN_NLL_vals_ctg, label=f"Profile $NLL(c_g$ min($c_{{tg}}$)) {profile_ctg_label}")
        plt.axhline(1, color='red', linestyle='--', label="68% CL ($2\\Delta NLL = 1$)")
        plt.xlabel(r"Wilson coefficient $c_{tg}$")
        plt.ylabel("2$\\Delta$NLL")
        plt.ylim(0, 10)
        plt.legend()
        plt.grid()
    
    
        # Minimized c_g for c_tg scan
        plt.subplot(2, 2, 4)
        plt.plot(ctg_values, minimized_cg_for_ctg, label=r"Minimised $c_{g}$ for each $c_{tg}$")
        plt.xlabel(r"Wilson coefficient $c_{tg}$")
        plt.ylabel(r"Minimised $c_{g}$")
        plt.legend()
        plt.grid()
    
        plt.tight_layout()
        plt.show()    

    return {
        'cg_values': cg_values,
        'ctg_values': ctg_values,
        'frozen_NN_NLL_vals_cg': frozen_NN_NLL_vals_cg,
        'profile_NN_NLL_vals_cg': profile_NN_NLL_vals_cg,
        'minimized_ctg_for_cg': minimized_ctg_for_cg,
        'frozen_NN_NLL_vals_ctg': frozen_NN_NLL_vals_ctg,
        'profile_NN_NLL_vals_ctg': profile_NN_NLL_vals_ctg,
        'minimized_cg_for_ctg': minimized_cg_for_ctg,
        'frozen_cg_label': frozen_cg_label,
        'profile_cg_label': profile_cg_label,
        'frozen_ctg_label': frozen_ctg_label,
        'profile_ctg_label': profile_ctg_label,
        'order': order
    }
      

def compare_frozen_scans(*datasets):
    """
    Compare 'frozen' scans (c_tg set to 0 when scanning over c_g, or c_g set to 0
    when scanning over c_tg). Accepts any number of data dictionaries.
    
    Each dictionary can contain:
      - "Name": a string to identify the data set in the legend (optional).
      - "cg_values" and "ctg_values": arrays of the scanned Wilson coefficients.
      - Keys for 'frozen' results, e.g.:
          "frozen_NN_NLL_vals_cg", "frozen_NN_NLL_vals_ctg"
          "frozen_chi_squared_cg", "frozen_chi_squared_ctg"
      - Corresponding label keys, e.g.:
          "frozen_cg_label", "frozen_ctg_label"
    """
    # Prepare figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    fig.suptitle("Comparison of FROZEN Scans (NLL vs. Chi-Squared) for Multiple Data Sets")

    # -- Left subplot: c_g scan (frozen c_tg) --
    ax_left = axes[0]
    left_keys = [
        ("frozen_NN_NLL_vals_cg",    "NN NLL (frozen c_tg)",    "frozen_cg_label"),
        ("frozen_chi_squared_cg",    r"STXS $\chi^2$ (frozen c_tg)", "frozen_cg_label"),
    ]
    
    for data in datasets:
        cg_values = data["cg_values"]  # The x-values for the c_g scan
        dataset_name = data.get("Name", "")  # Might be empty if not provided

        for data_key, method_label, dict_label_key in left_keys:
            if data_key in data:
                user_label = data.get(dict_label_key, "")
                
                # Build the legend label, including the Name only if it exists
                if dataset_name:
                    legend_label = f"{dataset_name} - {method_label}"
                else:
                    legend_label = method_label
                
                # Add user_label if it exists
                if user_label:
                    legend_label += f" {user_label}"

                ax_left.plot(cg_values, data[data_key], label=legend_label, lw=2)

    # Horizontal lines for confidence regions
    ax_left.axhline(1.0, color="red", linestyle="--", label="68% CL (2ΔNLL = 1)")
    ax_left.axhline(4.0, color="blue", linestyle="--", label="95% CL (2ΔNLL = 4)")
    
    ax_left.set_ylim(0, 10)
    ax_left.set_xlabel(r"$c_g$")
    ax_left.set_ylabel("2ΔNLL or Δχ²")
    ax_left.legend()
    ax_left.grid(True)

    # -- Right subplot: c_tg scan (frozen c_g) --
    ax_right = axes[1]
    right_keys = [
        ("frozen_NN_NLL_vals_ctg",    "NN NLL (frozen c_g)",    "frozen_ctg_label"),
        ("frozen_chi_squared_ctg",    r"STXS $\chi^2$ (frozen c_g)", "frozen_ctg_label"),
    ]

    for data in datasets:
        ctg_values = data["ctg_values"]  # The x-values for the c_tg scan
        dataset_name = data.get("Name", "")

        for data_key, method_label, dict_label_key in right_keys:
            if data_key in data:
                user_label = data.get(dict_label_key, "")

                if dataset_name:
                    legend_label = f"{dataset_name} - {method_label}"
                else:
                    legend_label = method_label
                
                if user_label:
                    legend_label += f" {user_label}"

                ax_right.plot(ctg_values, data[data_key], label=legend_label, lw=2)

    # Horizontal lines for confidence regions
    ax_right.axhline(1.0, color="red", linestyle="--", label="68% CL (2ΔNLL = 1)")
    ax_right.axhline(4.0, color="blue", linestyle="--", label="95% CL (2ΔNLL = 4)")

    ax_right.set_ylim(0, 10)
    ax_right.set_xlabel(r"$c_{tg}$")
    ax_right.set_ylabel("2ΔNLL or Δχ²")
    ax_right.legend()
    ax_right.grid(True)

    plt.tight_layout()
    plt.show()


def compare_profile_scans(*datasets):
    """
    Compare 'profile' scans (profiling over c_tg when scanning c_g, or profiling
    over c_g when scanning c_tg). Accepts any number of data dictionaries.
    
    Each dictionary can contain:
      - "Name": a string to identify the data set in the legend (optional).
      - "cg_values" and "ctg_values": arrays of the scanned Wilson coefficients.
      - Keys for 'profile' results, e.g.:
          "profile_NN_NLL_vals_cg", "profile_NN_NLL_vals_ctg"
          "profile_chi_squared_cg", "profile_chi_squared_ctg"
      - Corresponding label keys, e.g.:
          "profile_cg_label", "profile_ctg_label"
    """
    # Prepare figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    fig.suptitle("Comparison of PROFILE Scans (NLL vs. Chi-Squared) for Multiple Data Sets")

    # -- Left subplot: c_g profile scan (profiling over c_tg) --
    ax_left = axes[0]
    left_keys = [
        ("profile_NN_NLL_vals_cg",    "NN NLL (profiled over c_tg)",    "profile_cg_label"),
        ("profile_chi_squared_cg",    r"STXS $\chi^2$ (profiled over c_tg)", "profile_cg_label"),
    ]
    
    for data in datasets:
        cg_values = data["cg_values"]
        dataset_name = data.get("Name", "")

        for data_key, method_label, dict_label_key in left_keys:
            if data_key in data:
                user_label = data.get(dict_label_key, "")
                
                if dataset_name:
                    legend_label = f"{dataset_name} - {method_label}"
                else:
                    legend_label = method_label
                
                if user_label:
                    legend_label += f" {user_label}"

                ax_left.plot(cg_values, data[data_key], label=legend_label, lw=2)

    # Confidence lines
    ax_left.axhline(1.0, color="red", linestyle="--", label="68% CL (2ΔNLL = 1)")
    ax_left.axhline(4.0, color="blue", linestyle="--", label="95% CL (2ΔNLL = 4)")

    ax_left.set_xlabel(r"$c_g$")
    ax_left.set_ylabel("2ΔNLL or Δχ²")
    ax_left.legend()
    ax_left.set_ylim(0, 10)
    ax_left.grid(True)

    # -- Right subplot: c_tg profile scan (profiling over c_g) --
    ax_right = axes[1]
    right_keys = [
        ("profile_NN_NLL_vals_ctg",    "NN NLL (profiled over c_g)",    "profile_ctg_label"),
        ("profile_chi_squared_ctg",    r"STXS $\chi^2$ (profiled over c_g)", "profile_ctg_label"),
    ]
    
    for data in datasets:
        ctg_values = data["ctg_values"]
        dataset_name = data.get("Name", "")

        for data_key, method_label, dict_label_key in right_keys:
            if data_key in data:
                user_label = data.get(dict_label_key, "")
                
                if dataset_name:
                    legend_label = f"{dataset_name} - {method_label}"
                else:
                    legend_label = method_label
                
                if user_label:
                    legend_label += f" {user_label}"

                ax_right.plot(ctg_values, data[data_key], label=legend_label, lw=2)

    # Confidence lines
    ax_right.axhline(1.0, color="red", linestyle="--", label="68% CL (2ΔNLL = 1)")
    ax_right.axhline(4.0, color="blue", linestyle="--", label="95% CL (2ΔNLL = 4)")

    ax_right.set_xlabel(r"$c_{tg}$")
    ax_right.set_ylabel("2ΔNLL or Δχ²")
    ax_right.legend()
    ax_right.set_ylim(0, 10)
    ax_right.grid(True)

    plt.tight_layout()
    plt.show()

def weighted_quantile(values, quantiles, weights):
    """
    Compute weighted percentiles
    
    Parameters:
    values: array-like of values
    quantiles: array-like of quantiles to compute (between 0 and 1)
    weights: array-like of weights for each value
    
    Returns:
    array-like of weighted percentiles
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    weights = np.array(weights)
    
    # Sort values and weights
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    # Calculate cumulative weights
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    
    # Interpolate
    return np.interp(quantiles, weighted_quantiles, values)
    