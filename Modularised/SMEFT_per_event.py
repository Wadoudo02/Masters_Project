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


plot_entire_chain = True

# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF x 10", "red"],
    "VH" : ["VH x 10", "orange"],
    #"Data" : ["Data", "green"]
}

def add_SMEFT_weights(proc_data, cg, ctg, name = "new_weights", quadratic = False):
    proc_data[name] = proc_data['plot_weight'] * (1 + proc_data['a_cg'] * cg + proc_data['a_ctgre'] * ctg)
    if quadratic:
        proc_data[name] += (cg ** 2) * proc_data["b_cg_cg"]  + cg * ctg * proc_data["b_cg_ctgre"]  + (ctg ** 2) * proc_data["b_ctgre_ctgre"]
    return proc_data

#%%

cg = 0.15
ctg = 0.22

quadratic_order = True

plot_size = (12, 8)

# Load dataframes
dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]
    '''
    if proc == "Data":
        dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']
        
        bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
        labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
        dfs[proc]['category'] = pd.cut(dfs[proc]['pt_sel'], bins=bins, labels=labels, right=False)
        
        continue
    '''

    # Reweight to target lumi
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)
    
    if proc == "ttH":
        add_SMEFT_weights(dfs[proc], cg, ctg,"plot_weight", quadratic_order)

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
    
    
    mask = dfs[proc]['n_jets_sel'] >= 2
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.7)
    mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    # Categorise events: separate regions of high EFT enhancement vs low EFT enhancement

    # Categorise events: separate into 5 categories by pt
    bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
    labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
    dfs[proc]['category'] = pd.cut(dfs[proc]['pt_sel'], bins=bins, labels=labels, right=False)
    
    '''
    conditions = [
    (dfs[proc]['pt/mass']*mass < 60),      # Category 0: low p_T
    (dfs[proc]['pt/mass']*mass >= 60) & (dfs[proc]['pt/mass']*mass < 120),  # Category 1: medium p_T
    (dfs[proc]['pt/mass']*mass >= 120) & (dfs[proc]['pt/mass']*mass < 200), # Category 2: higher p_T
    (dfs[proc]['pt/mass']*mass >= 200) & (dfs[proc]['pt/mass']*mass < 300), # Category 3: even higher p_T
    (dfs[proc]['pt/mass']*mass >= 300)     # Category 4: highest p_T
    ]
    #dfs[proc]['category'] = np.array(dfs[proc]['n_leptons'] >= 1, dtype='int')
    dfs[proc]['category'] = np.select(conditions, [0, 1, 2, 3, 4])
    print(np.select(conditions, [0,1,2,3,4]))
      '''  
# Extract different cat integers
cats_unique = []
for proc in procs.keys():
    for cat in np.unique(dfs[proc]['category']):
        if cat not in cats_unique:
            cats_unique.append(cat)
  
cats_unique = labels.copy()

def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 100))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution for all categories in a single vertically stacked figure

background_estimates = {}

mass_range = (120, 130)  # Needed here to get BG estimate
mass_bins = 5

v = "mass"
v_dfs = v + "_sel"
n_categories = len(cats_unique)
fig, axs = plt.subplots(nrows=n_categories, ncols=1, figsize=(15, 5 * n_categories), sharex=True, dpi = 300)

for i, cat in enumerate(cats_unique):
    ax = axs[i] if n_categories > 1 else axs
    print(f" --> Processing: {v} in category {cat}")
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
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # Perform curve fitting, ignoring empty bins (where counts == 0)
            non_zero_indices = counts > 0
            popt, pcov = curve_fit(exponential_decay, bin_centers[non_zero_indices], counts[non_zero_indices])
            A, lambd = popt  # Unpack fitted parameters

            # Background estimate
            BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
            bin_estimates = []
            for j in range(len(BG_estimate_bin_edges) - 1):
                integral, _ = quad(exponential_decay, BG_estimate_bin_edges[j], BG_estimate_bin_edges[j + 1], args=(A, lambd))
                bin_estimates.append(integral)

            print(f"Background estimates for category {cat}: {bin_estimates}")

            # Store the result
            if cat not in background_estimates:
                background_estimates[cat] = {}
            background_estimates[cat][proc] = bin_estimates

        # Plot histograms
        print(f" --> Plotting: {v} in category {cat}")
        counts, bin_edges, _ = ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

        if proc == "background":
            # Plot the fitted exponential decay curve
            x_fit = np.linspace(xrange[0], xrange[1], 1000)
            y_fit = exponential_decay(x_fit, A, lambd)
            ax.plot(x_fit, y_fit, color="red", linestyle="--",
                    label=f"Exponential Fit\n$A={A:.2f}$, $\\lambda={lambd:.4f}$")

    ax.set_ylabel("Events")
    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best', ncol=2)

    hep.cms.label(f"category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

# Add shared x-axis label for all subplots
axs[-1].set_xlabel(sanitized_var_name)

# Add a grand title to the entire figure
fig.suptitle(f"Diphoton Mass Distribution Across Categories, $c_g = {cg}, c_{{tg}} = {ctg}$ ", fontsize=30, y=0.95)

# Adjust layout to fit all subplots neatly
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{plot_path}/{v}_all_categories.png", bbox_inches="tight")
plt.show()
    

                                                                                                                                                                                                              


