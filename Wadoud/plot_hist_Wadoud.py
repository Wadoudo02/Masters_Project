import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *

# Constants

total_lumi = 7.9804

# Normalise plots to unity: compare shapes
plot_fraction = bool(int(sys.argv[2]))

# Plot size: customizable figure size (default: 10x8)
plot_size = (12, 8)

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF", "red"],
    "VH" : ["VH", "orange"]
}

# Vars to plot
vars_to_plot = sys.argv[1].split(",")

# Load dataframes
dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    #if i==0:
        #print(" --> Columns in first dataframe: ", list(dfs[proc].columns))

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass'] == dfs[proc]['mass'])]

    # Add additional variables
    # Example: (second-)max-b-tag score
    b_tag_scores = np.array(dfs[proc][['j0_btagB', 'j1_btagB', 'j2_btagB', 'j3_btagB']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    # Add nans back in for plotting tools below
    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score'] = second_max_b_tag_score

# Create a customizable figure
fig, ax = plt.subplots(1, 1, figsize=plot_size)

for v in vars_to_plot:
    print(f" --> Plotting: {v}")
    nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
    # Loop over procs and add histogram
    for proc in procs.keys():
        label, color = procs[proc]
        if plot_fraction:
            label = label.split(" ")[0]

        x = np.array(dfs[proc][v])
        
        # Remove nans from array and calculate fraction of events which have real value
        mask = x == x
        pc = 100 * mask.sum() / len(x)
        if pc != 100:
            # Add information to label
            label += f", {pc:.1f}% plotted"
        x = x[mask]

        # Event weight
        w = np.array(dfs[proc]['plot_weight'])[mask]
        if plot_fraction:
            w /= w.sum()

        ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

    ax.set_xlabel(sanitized_var_name)
    if plot_fraction:
        ax.set_ylabel("Fraction of events")
    else:
        ax.set_ylabel("Events")
    
    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best')
    
    hep.cms.label("", year="2022 (preEE)", com="13.6", lumi=total_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    ext = "_norm" if plot_fraction else ""

    if '/' in v:
        v = v.replace('/', '_div_')

    #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
    fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
    ax.cla()
