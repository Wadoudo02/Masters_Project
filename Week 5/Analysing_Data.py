#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:05:47 2024

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit

plt.style.use(hep.style.CMS)

from utils_W5 import *

Raw_Data = pd.read_parquet(f"{sample_path}/Data_processed_selected.parquet")
Raw_Data = Raw_Data[(Raw_Data['mass_sel'] == Raw_Data['mass_sel'])]

Raw_Data['pt_sel'] = Raw_Data['pt-over-mass_sel'] * Raw_Data['mass_sel']

bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
Raw_Data['category'] = pd.cut(Raw_Data['pt_sel'], bins=bins, labels=labels, right=False)

cats_unique = ['0-60', '120-200', '200-300', '60-120', '>300']

fig, ax = plt.subplots(1,1, figsize=plot_size)
v = "mass_sel"

for cat in cats_unique:
    print(f" --> Plotting: {v} in cat{cat}")
    nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
    
    cat_mask = Raw_Data['category'] == cat
    x = np.array(Raw_Data[v][cat_mask])
    w = np.array(Raw_Data['plot_weight'][cat_mask])
    
    # Plot the histogram
    counts, bin_edges, _ = ax.hist(x, nbins, xrange, label="Raw Data", histtype='step', weights=w, edgecolor="green", lw=2)
    ax.set_xlabel(sanitized_var_name)
    ax.set_ylabel("Events")
    
    # Define the exponential decay function
    def exponential_decay(x, A, lambd):
        return A * np.exp(-lamb * (x - 100))
    
    # Calculate bin centers for fitting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Perform curve fitting, ignoring empty bins (where counts == 0)
    non_zero_indices = counts > 0
    popt, pcov = curve_fit(exponential_decay, bin_centers[non_zero_indices], counts[non_zero_indices])
    A, lambd = popt  # Unpack fitted parameters
    
    # Plot the fitted exponential decay curve
    x_fit = np.linspace(xrange[0], xrange[1], 1000)
    y_fit = exponential_decay(x_fit, A, lambd)
    ax.plot(x_fit, y_fit, color="red", linestyle="--", 
            label=f"Exponential Fit\n$A={A:.2f}$, $\\lambda={lambd:.4f}$")
    
    # Log scale if needed
    if is_log_scale:
        ax.set_yscale("log")
    
    # Add legend and labels
    ax.legend(loc='best')
    hep.cms.label(f"category_{cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
    
    # Save and show the plot
    plt.tight_layout()
    ext = f"Raw_Data_Fit_{cat}"
    fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
    ax.cla()
    plt.show()
    
    # Output the fitted parameters for reference
    print("Fitted parameters (A, Lambda):", popt)
    
    
#%%

Raw_Data = pd.read_parquet(f"{sample_path}/Data_processed_selected.parquet")
Raw_Data = Raw_Data[(Raw_Data['mass_sel'] == Raw_Data['mass_sel'])]
fig, ax = plt.subplots(1,1, figsize=plot_size)
v = "mass_sel"

nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
x = np.array(Raw_Data[v])
w = np.array(Raw_Data['plot_weight'])

# Plot the histogram
counts, bin_edges, _ = ax.hist(x, nbins, xrange, label="Raw Data", histtype='step', weights=w, edgecolor="green", lw=2)
ax.set_xlabel(sanitized_var_name)
ax.set_ylabel("Events")

# Define the exponential decay function
def exponential_decay(x, a, b):
    return a * np.exp(-b * (x-100))

# Calculate bin centers for fitting
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Perform curve fitting, ignoring empty bins (where counts == 0)
non_zero_indices = counts > 0
popt, pcov = curve_fit(exponential_decay, bin_centers[non_zero_indices], counts[non_zero_indices])
A, lambd = popt

# Plot the fitted exponential decay curve
x_fit = np.linspace(xrange[0], xrange[1], 1000)
y_fit = exponential_decay(x_fit, *popt)
ax.plot(x_fit, y_fit, color="red", linestyle="--", label=f"Exponential Fit\n$A={A:.2f}$, $\\lambda={lambd:.4f}$")

# Log scale if needed
if is_log_scale:
    ax.set_yscale("log")

# Add legend and labels
ax.legend(loc='best')
hep.cms.label("category", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

# Save and show the plot
plt.tight_layout()
ext = "Raw_Data_Fit"
fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
ax.cla()
plt.show()

# Output the fitted parameters for reference
print("Fitted parameters (a, b):", popt)