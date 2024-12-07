#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:18:27 2024

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from utils import *

plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300
mass_range = (120, 130)  # Mass range for histogram
mass_bins = 20  # Number of bins in the mass range
cg_ctg_pairs = [(0, 0), (0.15, 0), (0, 0.22), (0.15, 0.22)]  # SMEFT parameter pairs
pt_bins = [0, 60, 120, 200, 300, np.inf]
pt_labels = ['0-60', '60-120', '120-200', '200-300', '>300']

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

# Load and preprocess ttH data
print(" --> Loading process: ttH")
df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
df_tth = df_tth[(df_tth['mass_sel'] == df_tth['mass_sel'])]  # Remove NaNs
df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
df_tth['true_weight'] = df_tth['plot_weight'] / 10  # Remove x10 multiplier
df_tth['pt_sel'] = df_tth['pt-over-mass_sel'] * df_tth['mass_sel']

# Categorize events into pt categories
df_tth['category'] = pd.cut(df_tth['pt_sel'], bins=pt_bins, labels=pt_labels, right=False)

# Create figure with one subplot per pt category
fig, axs = plt.subplots(nrows=len(pt_labels), figsize=(15, 5 * len(pt_labels)), sharex=True, dpi=300)

for i, pt_cat in enumerate(pt_labels):
    ax = axs[i] if len(pt_labels) > 1 else axs
    print(f" --> Processing: pt category {pt_cat}")

    # Mask for current pt category
    pt_mask = df_tth['category'] == pt_cat

    # Overlay histograms for each SMEFT parameter pair
    for cg, ctg in cg_ctg_pairs:
        # Apply SMEFT weights
        df_tth_temp = df_tth.copy()
        df_tth_temp = add_SMEFT_weights(df_tth_temp, cg, ctg, name="plot_weight", quadratic=True)

        # Histogram data
        x = np.array(df_tth_temp['mass_sel'][pt_mask])
        w = np.array(df_tth_temp['plot_weight'][pt_mask])

        # Plot histogram
        ax.hist(
            x, bins=mass_bins, range=mass_range, weights=w,
            histtype='step', label=f"$(c_g, c_{{tg}}) = ({cg}, {ctg})$"
        )

    # Label and formatting
    ax.set_ylabel("Events")
    hep.cms.label(f"$p_T$ category: {pt_cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
    ax.legend(loc="best", ncol=2)

# Shared x-axis label
axs[-1].set_xlabel("Diphoton Mass (GeV)")

# Add a title
fig.suptitle("Diphoton Mass Distribution for ttH Across SMEFT Parameter Pairs", fontsize=20, y=0.95)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
fig.savefig(f"{plot_path}/ttH_SMEFT_pt_categories.png", bbox_inches="tight")
plt.show()