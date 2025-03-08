#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:16:13 2025

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



plot_entire_chain = True

plot_fraction = False

# Constants
total_lumi = 7.9804
target_lumi = 300

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

plot_size = (11, 6)

Quadratic = True

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
        #dfs[proc]['plot_weight'] *= 2
    else:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']
    
   # if proc == "ttH":
        #dfs[proc]['plot_weight'] *= 5
        

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
    
    
    mask = dfs[proc]['n_jets_sel'] >= 3
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.7)
    mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    if proc == "ttH_SMEFT":
        dfs[proc] = add_SMEFT_weights(dfs[proc], cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)






def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 120))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category

background_estimates = {}

mass_range = (120, 130)  # Needed here to get BG estimate
mass_bins = 5

v = "mass"
v_dfs = v + "_sel"

nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]

# Loop over procs and add histogram
for proc in procs.keys():
    label, color = procs[proc]

    x = np.array(dfs[proc][v_dfs])

    # Event weight
    w = np.array(dfs[proc]['true_weight'])

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



        background_estimates[proc] = bin_estimates

# Only plot if plot_entire_chain is True
if plot_entire_chain:
    fig, ax = plt.subplots(1, 1, figsize=plot_size)

    for proc in procs.keys():
        label, color = procs[proc]

        x = np.array(dfs[proc][v_dfs])

        # Event weight
        w = np.array(dfs[proc]['plot_weight'])
        

        counts, bin_edges, _ = ax.hist(x, nbins, xrange, density = plot_fraction, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

        if proc == "background":
            # Plot the fitted exponential decay curve
            x_fit = np.linspace(xrange[0], xrange[1], 1000)
            y_fit = exponential_decay(x_fit, A, lambd)
            ax.plot(x_fit, y_fit, color="red", linestyle="--",
                    label=f"Exponential Fit") # \n$A={A:.2f}$, $\\lambda={lambd:.4f}$

    ax.set_xlabel(sanitized_var_name)
    ax.set_ylabel("Events")

    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best', ncol=2)

    hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    
    plt.savefig(thesis_plot_path + "/all_procs.pdf")
    
    plt.show()
