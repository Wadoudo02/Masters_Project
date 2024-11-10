import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *

# Constants
total_lumi = 7.9804
target_lumi = 300
mass = 125

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF (x10)", "green"],
    "VH" : ["VH (x10)", "brown"]
}

cats = {0: "0-60",
        1: "60-120",
        2: "120-200",
        3: "200-300",
        4: "300-"}

col_name = "_sel"

# Load dataframes

dfs = {}
for i, proc in enumerate(procs.keys()):
    #if proc != "ttH": continue
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass'+col_name] == dfs[proc]['mass'+col_name])]

    # Reweight to target lumi 
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)

    # Calculate true weight: remove x10 multiplier for signal
    if proc in ['ggH', 'VBF', 'VH', 'ttH']:
        dfs[proc]['true_weight'+col_name] = dfs[proc]['plot_weight']/10
    else:
        dfs[proc]['true_weight'+col_name] = dfs[proc]['plot_weight']

    # Add variables
    # Example: (second-)max-b-tag score
    b_tag_scores = np.array(dfs[proc][['j0_btagB'+col_name, 'j1_btagB'+col_name, 'j2_btagB'+col_name, 'j3_btagB'+col_name]])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    # Add nans back in for plotting tools below
    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score'] = second_max_b_tag_score
    
    
    # Apply selection: separate ttH from backgrounds + other H production modes
    yield_before_sel = dfs[proc]['true_weight'+col_name].sum()
    #print("HT pre cuts:", len(dfs[proc]["HT"]))
    mask = dfs[proc]['n_jets'+col_name] >= 2
   # mask = mask & (dfs[proc]['HT'+col_name] > 200)
    mask = mask & (dfs[proc]['max_b_tag_score'] > 0.7)
    mask = mask & (dfs[proc]['second_max_b_tag_score'] > 0.4)
    #print("HT post cuts:", len(dfs[proc]["HT"][mask]))
    #mask = mask & dfs[proc]['minDeltaPhiJMET']<1
    #mask = mask & dfs[proc]['delta_eta_jj']<4
    #mask = mask& dfs[proc]['j3_pt']>10
    #mask = mask & (dfs[proc]['j1_eta']<=1) & (dfs[proc]['j1_eta']>=-1)
    #mask = mask & dfs[proc]['delta_phi_gg']>0.9
    #mask = mask & dfs[proc]['n_leptons'] >= 1
    #exit(0)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'+col_name].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")
    mass = dfs[proc]["mass"+col_name]
    pt_mass = dfs[proc]["pt-over-mass"+col_name]
    pt = pt_mass*mass
    #print(dfs[proc]["mass"])
    # Categorise events: separate regions of high EFT enhancement vs low EFT enhancement
    # e.g. number of leptons
    #conditions = get_pt_cat(pt)
    #dfs[proc]['category'] = np.array(dfs[proc]['n_leptons'] >= 1, dtype='int')
    dfs[proc]['category'] =  get_pt_cat(pt)
    
# Extract different cat integers
cats_unique = []
for proc in procs.keys():
    for cat in np.unique(dfs[proc]['category']):
        if cat not in cats_unique:
            cats_unique.append(cat)
print("Categories", np.unique(dfs[proc]['category']))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category
fig, ax = plt.subplots(1,1)
v = "mass"+col_name
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

    hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    ext = f"_cat{cat}"
    #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
    fig.savefig(f"{analysis_path}/{v}{ext}.png", bbox_inches="tight")
    ax.cla()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simple binned likelihood fit to mass histograms in signal window (120,130)
hists = {}
mass_range = (120,130)
mass_bins = 5
v = 'mass'+col_name

for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        cat_mask = dfs[proc]['category'] == cat
        hist_counts = np.histogram(dfs[proc][cat_mask][v], mass_bins, mass_range, weights=dfs[proc][cat_mask]['true_weight'+col_name])[0]
        
        hist_counts[-2] += hist_counts[-1]  # Add last bin to the second last
        hist_counts = hist_counts[:-1]      # Remove the last bin
        hists[cat][proc] = hist_counts
# Calculate NLL as a function of ttH signal strength (assuming fixed bkg and ggH yields)
NLL_vals = []
mu_vals = np.linspace(0,3,100)
new_samples = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
conf_matrix =get_conf_mat(new_samples)
#print(conf_matrix)
#print(hists)
for cat in cats_unique:
    cat_vals = []
    for mu in mu_vals:
        cat_vals.append(calc_NLL(hists, mu, conf_matrix[1], cat))
    print("    NLL vals for cat: ",cat_vals)
    NLL_vals.append(cat_vals)
    
# Plot NLL curve
for cat in cats_unique:
    #Best fit vals for category cat.
    vals = find_crossings((mu_vals,TwoDeltaNLL(NLL_vals[cat])),1.)
    label = add_val_label(vals)

    print(" --> Plotting 2NLL curve")
    fig, ax = plt.subplots()
    ax.plot(mu_vals, TwoDeltaNLL(NLL_vals[cat]), label=label)
    ax.axvline(1., label="SM (expected)", color='black', alpha=0.5)
    ax.axhline(1, color='grey', alpha=0.5, ls='--')
    ax.axhline(4, color='grey', alpha=0.5, ls='--')
    ax.set_ylim(0,8)
    ax.legend(loc='best')
    ax.set_xlabel("$\\mu_{ttH}$")
    ax.set_ylabel("q = 2$\\Delta$NLL")
    ax.set_title(f"Best fit for cat: {cats[cat]}")
    plt.tight_layout()
    #fig.savefig(f"{analysis_path}/2nll_vs_mu.pdf", bbox_inches="tight")
    fig.savefig(f"{analysis_path}/2nll_vs_mu_{cats[cat]}.png", bbox_inches="tight")
    ax.cla()
