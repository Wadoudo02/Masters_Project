import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *

# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF", "red"],
    "VH" : ["VH", "orange"]
}

plot_size = (12, 8)

# Load dataframes
dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass'] == dfs[proc]['mass'])]

    # Reweight to target lumi
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)

    # Calculate true weight: remove x10 multiplier for signal
    if proc in ['ggH', 'VBF', 'VH', 'ttH']:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']/10
    else:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']

    # Add variables
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
    
    # Apply selection: separate ttH from backgrounds + other H production modes
    yield_before_sel = dfs[proc]['true_weight'].sum()
    mask = dfs[proc]['n_jets'] >= 4
    mask = mask & dfs[proc]['max_b_tag_score'] > 0.8
    mask = mask & dfs[proc]['second_max_b_tag_score'] > 0.4
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")


    # Categorise events: separate regions of high EFT enhancement vs low EFT enhancement
    # e.g. number of leptons
    dfs[proc]['category'] = np.array(dfs[proc]['n_leptons'] >= 1, dtype='int')
    
# Extract different cat integers
cats_unique = []
for proc in procs.keys():
    for cat in np.unique(dfs[proc]['category']):
        if cat not in cats_unique:
            cats_unique.append(cat)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category

vars_to_plot = ["mass","n_jets","lead_pt","Muo0_pt","MET_pt","HT","dijet_mass","deltaR","delta_phi_gg","dilepton_mass","n_leptons","pt/mass"]

for v in vars_to_plot:

    fig, ax = plt.subplots(1,1, figsize=plot_size)
    for cat in cats_unique:
        print(f" --> Plotting: {v} in cat{cat}")
        nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
        # Loop over procs and add histogram
        for proc in procs.keys():
            label, color = procs[proc]
    
            cat_mask = dfs[proc]['category']==cat
    
            x = np.array(dfs[proc][v][cat_mask])
    
            # Event weight
            w = np.array(dfs[proc]['plot_weight'])[cat_mask]
    
            ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)
    
        ax.set_xlabel(sanitized_var_name)
        ax.set_ylabel("Events")
    
        if is_log_scale:
            ax.set_yscale("log")
    
        ax.legend(loc='best')
    
        hep.cms.label(f"Category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
    
        plt.tight_layout()
        ext = f"_cat{cat}"
        
        if '/' in v:
            v = v.replace('/', '_div_')
        
        #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
        fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
        ax.cla()
        plt.show()
        
        if '_div_' in v:
            v = v.replace('_div_','/')
        


