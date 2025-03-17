#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:23:33 2025

@author: wadoudcharbak
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:16:13 2025

@author: wadoudcharbak
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from scipy.optimize import curve_fit
from scipy.integrate import quad

from utils import *
import torch
from NN_utils import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOGGLE: set to True if you only want to show ttH + the background *bin lines*;
# set to False if you want to add other processes, too.
plot_simplified = False

plot_entire_chain = True
plot_fraction = False

# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH"        : ["ttH x 10", "mediumorchid"],
    "ggH"        : ["ggH x 10", "cornflowerblue"],
    "VBF"        : ["VBF x 10", "red"],
    "VH"         : ["VH x 10", "orange"],
}

plot_size = (10, 6)

Quadratic = True

# Load dataframes
dfs = {}
for proc in procs.keys():
    
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans and negative weights
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]
    dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]

    # Normalise and reweight
    yield_weight = dfs[proc]["plot_weight"].sum()
    dfs[proc]["plot_weight"] /= dfs[proc]["plot_weight"].sum()
    dfs[proc]["plot_weight"] *= yield_weight
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight'] * (target_lumi / total_lumi)

    # "true_weight" for plotting
    if proc in ['ggH', 'VBF', 'VH', 'ttH']:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight'] / 10
    else:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']

    # Example b-tag logic
    b_tag_scores = np.array(dfs[proc][['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1 * np.sort(-1 * b_tag_scores, axis=1)[:,0]
    second_max_b_tag_score = -1 * np.sort(-1 * b_tag_scores, axis=1)[:,1]

    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score_sel'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score_sel'] = second_max_b_tag_score

    # Selection
    yield_before_sel = dfs[proc]['true_weight'].sum()
    mask = dfs[proc]['n_jets_sel'] >= 3
    mask &= (dfs[proc]['max_b_tag_score_sel'] > 0.7)
    mask &= (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    mask &= (dfs[proc]['HT_sel'] > 200)

    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel / yield_before_sel) * 100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    # Additional variable
    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

# Simple exponential decay for the BG fit
def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 120))

# We'll fit background between 120 and 130 in 5 bins
mass_range = (120, 130)
mass_bins = 5

v = "mass"
v_dfs = v + "_sel"

# Example plotting dict
vars_plotting_dict = {
    "mass": [40, (100, 140), False, r"$m_{\gamma\gamma}$ (GeV)"],
}
nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]

# Fit the background to get bin-by-bin estimates
background_estimates = {}
if "background" in dfs:
    x_bg = np.array(dfs["background"][v_dfs])
    w_bg = np.array(dfs["background"]['true_weight'])

    counts_bg, bin_edges_bg = np.histogram(x_bg, bins=nbins, range=xrange, weights=w_bg)
    bin_centers_bg = 0.5 * (bin_edges_bg[1:] + bin_edges_bg[:-1])

    non_zero_indices = counts_bg > 0
    popt, pcov = curve_fit(exponential_decay,
                           bin_centers_bg[non_zero_indices],
                           counts_bg[non_zero_indices])
    A, lambd = popt

    # Evaluate the integral in 5 bins from 120 to 130
    BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
    bin_estimates = []
    for i in range(len(BG_estimate_bin_edges) - 1):
        left_edge  = BG_estimate_bin_edges[i]
        right_edge = BG_estimate_bin_edges[i+1]
        integral, _ = quad(exponential_decay, left_edge, right_edge, args=(A, lambd))
        bin_estimates.append(integral)

    background_estimates["background"] = bin_estimates

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
if plot_entire_chain:
    fig, ax = plt.subplots(1, 1, figsize=plot_size)

    # Decide which processes to show
    if plot_simplified:
        procs_to_plot = ["ttH"]  # e.g. just show ttH plus BG lines
    else:
        procs_to_plot = list(procs.keys())     # show everything

    for proc in procs_to_plot:
        if proc not in dfs:
            continue

        # Skip the "background" histogram, because we only want flat lines
        if proc == "background":
            continue

        label, color = procs[proc]
        x = np.array(dfs[proc][v_dfs])
        w = np.array(dfs[proc]['plot_weight'])

        ax.hist(x,
                bins=nbins,
                range=xrange,
                density=plot_fraction,
                label=label,
                histtype='step',
                weights=w,
                edgecolor=color,
                lw=2)

    # Now, draw horizontal lines for the BG *bin estimates* instead of the hist
    if "background" in background_estimates:
        bin_edges = np.linspace(120, 130, mass_bins + 1)
        for i, est in enumerate(background_estimates["background"]):
            left  = bin_edges[i]
            right = bin_edges[i+1]
            # A flat line from left to right at y = est
            ax.hlines(y=est, xmin=left, xmax=right,
                      color='black', linestyle='-', linewidth=2,
                      label="BG Estimate" if i==0 else None)  
            # Only label the first bin so it doesn't repeat in the legend

    # Draw vertical lines to show the bin boundaries from 120 to 130
    for edge in np.linspace(120,130,mass_bins+1):
        ax.axvline(edge, color="gray", linestyle=":", alpha=0.7)

    # Optionally annotate the actual numeric estimate near each line
    if "background" in background_estimates:
        for i, est in enumerate(background_estimates["background"]):
            left_edge  = BG_estimate_bin_edges[i]
            right_edge = BG_estimate_bin_edges[i+1]
            x_mid = 0.5 * (left_edge + right_edge)
            # Just place the text slightly above the line
            ax.text(x_mid, est*1.02, f"{est:.2f}",
                    ha="center", va="bottom", color="black", fontsize=15)

    # Axis labels and limits
    ax.set_xlabel(sanitized_var_name)
    ax.set_ylabel("Events" if not plot_fraction else "Fraction")
    ax.set_xlim(115, 135)  # narrower x-range

    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best', ncol=1)
    hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    # Save figure
    # thesis_plot_path = "/where/to/save/"
    # plt.savefig(thesis_plot_path + "/BG_estimates_flatline.pdf")
    plt.show()
    
    
    
mass_bins = 5
v = 'mass'
v_dfs = v + "_sel"

#%%

# Initialize histogram data for each reconstructed category and process

hists = {}
for proc in procs.keys():
    # Apply mask to categorize events by reconstructed category
    if proc == "background":
        hists[proc] = np.array(background_estimates[proc])
    else:
        hists[proc] = np.histogram(
            dfs[proc][v_dfs], 
            mass_bins, 
            mass_range, 
            weights=dfs[proc]['true_weight']
        )[0]
        
        

procs = {
    "background" : ["BG Estimate", "black"],
    "ttH"        : ["ttH x 10", "mediumorchid"],
    "ggH"        : ["ggH x 10", "cornflowerblue"],
    "VBF"        : ["VBF x 10", "red"],
    "VH"         : ["VH x 10", "orange"],
}


mass_bins   = 5
mass_range  = (120, 130)                              # 120 -> 130 in 5 bins
bin_edges   = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)  # [120,122,124,126,128,130]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # midpoints
bin_width   = bin_edges[1] - bin_edges[0]             # should be 2.0

# TOGGLE: True => plot BG as black lines w/ text; False => plot BG as a bar
plot_bg_as_line = True

fig, ax = plt.subplots(figsize=plot_size, dpi = 300)

# Separate out the background from the signals
bg_proc = "background"
bg_label, bg_color = procs[bg_proc]
bg_vals = hists[bg_proc]

signals = [p for p in procs.keys() if p != "background"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) If plotting background as bar:
#    - We'll treat it just like signals, except no x10 factor.
# 2) If plotting background as line:
#    - We'll skip the bar entirely for background, and do hlines + text.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if plot_bg_as_line:
    # Just plot signals as grouped bars
    # ------------------------------------------------
    n_signals        = len(signals)
    group_bar_width  = 0.8 * bin_width
    single_bar_width = group_bar_width / n_signals

    for i, proc in enumerate(signals):
        label, color = procs[proc]
        y_vals = hists[proc].copy()
        y_vals *= 10  # scale signals by x10

        shift = (i - 0.5*(n_signals-1)) * single_bar_width
        ax.bar(
            bin_centers + shift,
            y_vals,
            single_bar_width,
            label=label ,  # so we don't repeat in legend
            color=color,
            alpha=0.7,
            edgecolor="black"
        )
    
    # Now draw background as horizontal lines + text
    # ------------------------------------------------
    for i in range(mass_bins):
        left_edge  = bin_edges[i]
        right_edge = bin_edges[i+1]
        y = bg_vals[i]

        # Flat black line
        ax.hlines(
            y=y, xmin=left_edge, xmax=right_edge,
            color=bg_color, linewidth=2,
            label=bg_label if i == 0 else "",  # label once in legend
            alpha=1.0
        )
        # Text with translucent white box
        x_text = 0.5 * (left_edge + right_edge)
        ax.text(
            x_text, y * 1.02,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=15,
            bbox=dict(
                boxstyle="square,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.7  # translucent
            )
        )

else:
    # Plot background as a bar along with signals
    # ------------------------------------------------
    # We'll just handle everything in a single loop, treating background or signals slightly differently
    all_procs = list(procs.keys())
    n_procs   = len(all_procs)

    group_bar_width  = 0.8 * bin_width
    single_bar_width = group_bar_width / n_procs

    for i, proc in enumerate(all_procs):
        label, color = procs[proc]
        y_vals = hists[proc].copy()

        # Multiply signals by 10, but not BG
        if proc != "background":
            y_vals *= 10
        
        shift = (i - 0.5*(n_procs-1)) * single_bar_width
        ax.bar(
            bin_centers + shift,
            y_vals,
            single_bar_width,
            label=label,  # show label only once
            color=color,
            alpha=0.7,
            edgecolor="black"
        )

# Draw vertical lines for bin edges
for edge in bin_edges:
    ax.axvline(edge, color='gray', linestyle=':', alpha=0.7)

# Cosmetic settings
ax.set_xlabel(r"$m_{\gamma\gamma}$ (GeV)")
ax.set_ylabel("Event Counts")
ax.set_xlim(119, 136)
ax.legend(ncol=1, loc='best')

hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

plt.tight_layout()
plt.show()