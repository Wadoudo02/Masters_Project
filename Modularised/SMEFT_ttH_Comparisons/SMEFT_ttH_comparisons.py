

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from utils import *

plt.style.use(hep.style.CMS)

plot_fraction = True

Quadratic = True

# Constants
total_lumi = 7.9804
target_lumi = 300
cg_ctg_pairs = [(0, 0),  (0.3, 0.69)]  # SMEFT parameter pairs
pt_bins = [0, 60, 120, 200, 300, np.inf]
pt_labels = ['0-60', '60-120', '120-200', '200-300', '>300']



def add_SMEFT_weights(proc_data, cg, ctg, quadratic = True):

    new_w = proc_data["true_weight"] * (1.0 + proc_data["a_cg"]*cg + proc_data["a_ctgre"]*ctg)
    # optional quadratic:
    if quadratic:
        new_w += proc_data["true_weight"] * ((cg**2)*proc_data["b_cg_cg"] + (cg*ctg)*proc_data["b_cg_ctgre"] + (ctg**2)*proc_data["b_ctgre_ctgre"])
    return new_w

# Variable to plot
v = "pt"

# Extract plotting details from vars_plotting_dict
if v == "pt":
    num_bins, plot_range, logplot, x_label = [50, (0, 1000), False, "$p_T$ [GeV]"]
else:
    num_bins, plot_range, logplot, x_label = vars_plotting_dict[v]

v += "_sel"

# Load and preprocess ttH data
print(" --> Loading process: ttH")
df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
df_tth = df_tth[(df_tth["mass_sel"] == df_tth["mass_sel"])]  # Remove NaNs in the selected variable
df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
df_tth['true_weight'] = df_tth['plot_weight'] / 10  # Remove x10 multiplier
df_tth['pt_sel'] = df_tth['pt-over-mass_sel'] * df_tth['mass_sel']

yield_weight = df_tth["true_weight"].sum()

invalid_weights = df_tth["true_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]
    
df_tth["true_weight"] /= df_tth["true_weight"].sum()
df_tth["true_weight"] *= yield_weight

# Add variables
# Example: (second-)max-b-tag score
b_tag_scores = np.array(df_tth[['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]


# Add nans back in for plotting tools below
max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
df_tth['max_b_tag_score_sel'] = max_b_tag_score
df_tth['second_max_b_tag_score_sel'] = second_max_b_tag_score

# Apply selection: separate ttH from backgrounds + other H production modes


mask = df_tth['n_jets_sel'] >= 0
mask = mask & (df_tth['max_b_tag_score_sel'] > 0.4)
#mask = mask & (df_tth['second_max_b_tag_score_sel'] > 0.4)
#mask = mask & (df_tth['HT_sel'] > 200)

df_tth = df_tth[mask]

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

# Define color palette
colors = sns.color_palette("husl", len(cg_ctg_pairs))

labels_weights = ["SM", "SMEFT"]

# Overlay histograms for each SMEFT parameter pair
for j, (cg, ctg) in enumerate(cg_ctg_pairs):
    # Apply SMEFT weights
    df_tth_temp = df_tth.copy()
    df_tth_temp["true_weight"] = add_SMEFT_weights(df_tth_temp, cg, ctg, quadratic=Quadratic)
    
    df_tth_temp["true_weight"] /= df_tth_temp["true_weight"].sum()
    df_tth_temp["true_weight"] *= 1e4
    

    # Histogram data
    x = np.array(df_tth_temp[v])
    w = np.array(df_tth_temp['true_weight'])

    if plot_fraction:
        w /= w.sum()

    print(len(w))
    print(w)

    # Plot histogram with color
    if len(cg_ctg_pairs) == 2:
        ax.hist(
                x, bins=num_bins, range=plot_range, density=False, weights=w,
                histtype='step', color=colors[j], linewidth=2, alpha=1, label=f"{labels_weights[j]} $(c_g, c_{{tg}}) = ({cg}, {ctg})$")
    else:
        ax.hist(
                x, bins=num_bins, range=plot_range, density=False, weights=w,
                histtype='step', color=colors[j], linewidth=2, alpha=1, label=f"$(c_g, c_{{tg}}) = ({cg}, {ctg})$")
    
# Label and formatting
ax.set_ylabel("Fraction of Events")
if logplot:
    ax.set_yscale("log")
#hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
ax.legend(loc="best", ncol=1)

'''
# Get the handles and labels
handles, labels = ax.get_legend_handles_labels()

# Swap the order of handles and labels
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]

# Create a new legend with swapped order
ax.legend(handles, labels)
'''

# Shared x-axis label from vars_plotting_dict
ax.set_xlabel(x_label)
# Define the boundaries and labels
boundaries = [60, 120, 200, 300]
pt_labels = ['0-60', '60-120', '120-200', '200-300', '>300']

# Draw vertical lines and add labels at the boundaries
for i, b in enumerate(boundaries):
    ax.axvline(b, color='grey', linestyle='--', linewidth=1)
    # Place the label above the line using the x-axis transform for the y position
    # Here we label the boundary to the right (e.g. 60 gets labelled as "60-120")
    ax.text(b + 25, 0.95, pt_labels[i+1], rotation=270, transform=ax.get_xaxis_transform(),
            ha='right', va='top', color='grey', fontsize=12)
# Optionally, label the left-most category at the left edge of the plot
ax.text(ax.get_xlim()[0] + 25, 0.95, pt_labels[0], rotation=270, transform=ax.get_xaxis_transform(),
        ha='right', va='top', color='grey', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


# Save figure
#plt.savefig(thesis_plot_path + "/STXS_Categorisation_Big.pdf")
plt.show()