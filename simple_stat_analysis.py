import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *
from background_dist import *
from nll import *

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
    dfs[proc]['category'] =  get_pt_cat(pt, bins=[0,60,120,200,300])
    
# Extract different cat integers
cats_unique = []
for proc in procs.keys():
    for cat in np.unique(dfs[proc]['category']):
        if cat not in cats_unique:
            cats_unique.append(cat)

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

#%%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simple binned likelihood fit to mass histograms in signal window (120,130)
print('''
NLL FROZEN SCAN USING INDIVIDUAL HISTOGRAMS
''')
hists = {}
mass_range = (120,130)
mass_bins = 5
v = 'mass'+col_name
for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        #Adding background events from background distribution
        if proc=="background":
            hist_counts=get_back_int(dfs["background"], cat, mass_range, mass_bins, len(cats_unique))
        else:
            cat_mask = dfs[proc]['category'] == cat
            hist_counts = np.histogram(dfs[proc][cat_mask][v], mass_bins, mass_range, weights=dfs[proc][cat_mask]['true_weight'+col_name])[0]
        
        # hist_counts[-2] += hist_counts[-1]  # Add last bin to the second last
        # hist_counts = hist_counts[:-1]      # Remove the last bin
        hists[cat][proc] = hist_counts
#print(hists)
# Calculate NLL as a function of ttH signal strength (assuming fixed bkg and ggH yields)
NLL_vals = []
mu_vals = np.linspace(-2,5,100)
new_samples = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
conf_matrix_raw, conf_matrix, conf_matrix_recon = get_conf_mat(new_samples) #conf_matrix[2] is the one normalised by recon
init_mu = [1 for i in range(len(conf_matrix))]
#%%
# '''
# Frozen scan over mu
# '''

# for cat in cats_unique:
#     cat_vals = []
#     for mu in mu_vals:
#         init_mu[cat]=mu
#         cat_vals.append(calc_NLL(hists, init_mu, conf_matrix))
#     #print(cat_vals)
#     init_mu[cat]=1
#     #print("________________________________")
#     #print(f"NLL vals for cat - {cat}: ",cat_vals)
#     NLL_vals.append(cat_vals)

# '''
# Plotting NLL curves
# '''
# fig, axes = plt.subplots(2, 3,figsize=(15, 10))
# all_mu = []
# # Plot NLL curve
# for idx in range(len(cats_unique)):
#     cat = cats_unique[idx]
#     #Best fit vals for category cat.
#     vals = find_crossings((mu_vals,TwoDeltaNLL(NLL_vals[cat])),1.)
#     all_mu.append(vals[0])
#     label = add_val_label(vals)
#     ax = axes[idx//3,idx%3]
#     print(" --> Plotting 2NLL curve")
#     ax.plot(mu_vals, TwoDeltaNLL(NLL_vals[cat]), label=label)
#     ax.axvline(1., label="SM (expected)", color='black', alpha=0.5)
#     ax.axhline(1, color='grey', alpha=0.5, ls='--')
#     ax.axhline(4, color='grey', alpha=0.5, ls='--')
#     ax.set_ylim(0,8)
#     ax.legend(loc='best')
#     ax.set_xlabel("$\\mu_{ttH}$")
#     ax.set_ylabel("q = 2$\\Delta$NLL")
#     ax.set_title(f"Best fit for cat: {cats[cat]}", fontsize=15)
#     #plt.tight_layout()
#     #fig.savefig(f"{analysis_path}/2nll_vs_mu.pdf", bbox_inches="tight")
#     #fig.savefig(f"{analysis_path}/2nll_vs_mu_{cats[cat]}.png", bbox_inches="tight")
#     #ax.cla()
# fig.delaxes(axes[1,2])
# fig.suptitle("Frozen NLL scan for each mu using ind hist")
# fig.savefig(f"{analysis_path}/2nll_vs_mu_subplot_fro.png", bbox_inches="tight")
# #fig.show()
#%%
'''
Scanning using combined hist
'''
print('''
NLL FROZEN SCAN USING COMBINED HISTOGRAM
''')
print(f'''Before combining
      {hists}
      ''')
comb_hist = build_combined_histogram(hists, conf_matrix, mass_bins=5)
#NLL_vals = []
print(f'''
After combining
      {comb_hist}
''')
bin_edges = np.arange(len(next(iter(comb_hist.values()))))
for cat, hist in comb_hist.items():
    plt.bar(bin_edges,hist, label=cat)
    plt.title(f"Cat: {cat}")
    plt.show()
init_mu = np.ones(len(conf_matrix))

fig, axes = plt.subplots(ncols=5, figsize=(25, 5), dpi = 300, sharex=True)

for i in range(len(init_mu)):
    NLL_vals = []
    for mu in mu_vals:
        init_mu[i] = mu  # Set the i-th \mu to the scan value
        # Calculate the NLL for the current set of \mu values
        NLL_vals.append(calc_NLL_comb(comb_hist, init_mu,signal='ttH'))

    vals = find_crossings((mu_vals, TwoDeltaNLL(NLL_vals)), 1.)
    init_mu[i] = vals[0]#[0]
    label = add_val_label(vals)
    
    # Plotting each NLL curve on a separate subplot
    ax = axes[i]
    ax.plot(mu_vals, TwoDeltaNLL(NLL_vals), label=label)
    ax.axvline(1., label="SM (expected)", color='green', alpha=0.5)
    ax.axhline(1, color='grey', alpha=0.5, ls='--')
    ax.axhline(4, color='grey', alpha=0.5, ls='--')
    ax.set_ylim(0, 8)
    ax.legend(loc='best')
    ax.set_ylabel("q = 2$\\Delta$NLL")
    ax.set_title(f"Optimising $\\mu_{i}$")
fig.suptitle("Frozen NLL scan for each mu using comb hist")
all_mu = init_mu
print("The optimised values of mu are:", init_mu)
#%%
#'''
# Wadoud profiled nll
# '''
# plot_entire_chain = True
# def calc_NLL(combined_histogram, mus, signal='ttH'):
#     """
#     Calculate the NLL using the combined 25-bin histogram with variable \mu parameters.

#     Parameters:
#         combined_histogram (dict): Combined histogram for each process across 25 bins.
#         mus (list or array): Signal strength modifiers, one for each truth category.
#         signal (str): The signal process (default 'ttH').

#     Returns:
#         float: Total NLL.
#     """
#     NLL_total = 0.0
#     num_bins = len(next(iter(combined_histogram.values())))  # Total bins (should be 25)

#     # Loop over each bin in the combined histogram
#     for bin_idx in range(num_bins):
#         expected_total = 0.0
#         observed_count = 0.0
#         #breakpoint()
#         for proc, yields in combined_histogram.items():
#             if signal in proc:
#                 # Extract the truth category index from the signal label, e.g., "ttH_0"
#                 truth_cat_idx = int(proc.split('_')[1])
#                 mu = mus[truth_cat_idx]  # Apply the appropriate \mu for this truth category
#                 expected_total += mu * yields[bin_idx]
#             else:
#                 expected_total += yields[bin_idx]

#             observed_count += yields[bin_idx]  # Observed count from all processes in this bin

#         # Avoid division by zero in log calculation
#         expected_total = max(expected_total, 1e-10)
        
#         # Calculate NLL contribution for this bin
#         NLL_total += observed_count * np.log(expected_total) - expected_total

#     return -NLL_total

# # Define the objective function for NLL
# def objective_function(mus, fixed_mu_index, fixed_mu_value):
#     """
#     Objective function to compute NLL for given mu values, fixing one mu.
    
#     Parameters:
#         mus (array-like): Array of 4 mu values to optimize.
#         fixed_mu_index (int): Index of the mu being scanned (fixed during optimization).
#         fixed_mu_value (float): The fixed value for the scanned mu.
    
#     Returns:
#         float: NLL value for the given set of mu values.
#     """
#     full_mus = np.insert(mus, fixed_mu_index, fixed_mu_value)  # Reconstruct full mu array
#     return calc_NLL(comb_hist, full_mus, signal='ttH')

# # Configuration
# mu_values = np.linspace(-2, 5, 100)  # Range for scanning a single mu
# mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]
# bounds = [(0, 3) for _ in range(4)]  # Bounds for the other mu parameters

# frozen_mus = mus_initial.copy()

# # Prepare the plots
# if plot_entire_chain:
#     fig, axes = plt.subplots(nrows=5, figsize=(8, 30), dpi=300, sharex=True)

# # Perform both frozen scan and profile scan
# for i in range(5):
#     frozen_NLL_vals = []
#     profile_NLL_vals = []
    
#     for mu in mu_values:
#         # Frozen scan: keep other mu values constant
        
#         frozen_mus[i] = mu
#         frozen_NLL_vals.append(calc_NLL(comb_hist, frozen_mus, signal='ttH'))
        
#         # Profile scan: optimize the other mu values
#         initial_guess = [mus_initial[j] for j in range(5) if j != i]
#         obj_func = lambda reduced_mus: objective_function(reduced_mus, fixed_mu_index=i, fixed_mu_value=mu)
#         result = minimize(obj_func, initial_guess, bounds=bounds, method='L-BFGS-B')
        
#         profile_NLL_vals.append(result.fun)
    
#     # Convert to 2ΔNLL
#     frozen_NLL_vals = TwoDeltaNLL(frozen_NLL_vals)
#     profile_NLL_vals = TwoDeltaNLL(profile_NLL_vals)
    
#     # Find crossings
#     frozen_vals = find_crossings((mu_values, frozen_NLL_vals), 1.)
#     profile_vals = find_crossings((mu_values, profile_NLL_vals), 1.)
#     frozen_label = add_val_label(frozen_vals)
#     profile_label = add_val_label(profile_vals)
    
#     # Keep optimal Frozen
    
#     frozen_mus[i] = frozen_vals[0]
    
#     if plot_entire_chain:
#     # Plotting each NLL curve on a separate subplot
#         ax = axes[i]
#         ax.plot(mu_values, frozen_NLL_vals, label=f"Frozen Scan: {frozen_label}", color='blue')
#         ax.plot(mu_values, profile_NLL_vals, label=f"Profile Scan: {profile_label}", color='red', linestyle='--')
#         ax.axvline(1., label="SM (expected)", color='green', alpha=0.5)
#         ax.axhline(1, color='grey', alpha=0.5, ls='--')
#         ax.axhline(4, color='grey', alpha=0.5, ls='--')
#         ax.set_ylim(0, 8)
#         ax.legend(loc='best')
#         ax.set_ylabel("q = 2$\\Delta$NLL")
#         ax.set_title(f"Optimising $\\mu_{i}$")

# if plot_entire_chain:
#     # Show the plot
#     plt.xlabel("$\\mu$ Value")
#     plt.tight_layout()
#     plt.show()

#%%
'''
Profiled fit for NLL 
'''
print('''
NLL PROFILED SCAN USING COMBINED HISTOGRAM
''')
best_mus = np.ones(5)
fig, ax = plt.subplots(1, 5, figsize=(40, 7))
for idx in range(len(conf_matrix)):
    best_mus[idx], nll_mu = profiled_NLL_fit(comb_hist,conf_matrix, idx, mu_vals)
    vals = find_crossings((mu_vals, TwoDeltaNLL(nll_mu)), 1.)
    print("Best mu for idx:", idx, "is", vals[0] ,"+/-", vals[1:])
    label = add_val_label(vals)
    ax[idx].plot(mu_vals, TwoDeltaNLL(nll_mu), label=label)
    ax[idx].set_title(f"NLL variation for mu idx: {idx}")
    ax[idx].set_ylabel("2 delta NLL")
    ax[idx].set_xlabel("mu")
    ax[idx].axvline(1., label="SM (expected)", color='green', alpha=0.5)
    ax[idx].axhline(1, color='grey', alpha=0.5, ls='--')
    ax[idx].axhline(4, color='grey', alpha=0.5, ls='--')
    ax[idx].set_ylim(0, 8)
    ax[idx].legend(loc='best')
fig.suptitle("Profiled NLL fit for each mu")
fig.savefig("nll_profiled_fit.png")
#%%
print('''
      NLL Global fit
''')

global_mu, global_nll, hessian, cov = global_nll_fit(comb_hist, conf_matrix)
print(f"""
      Global best fit for mu: {global_mu}
        Hessian: {hessian}
        Covariance: {cov}
""")


#%%
print('''
Getting Hessian matrix
''')
print("All mus:", all_mu)
hessian = np.zeros((len(conf_matrix), len(conf_matrix)))
#hessian = [[0 for i in range(5)] for j in range(5)]
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        hessian[i][j] = get_hessian(i,j, hists, all_mu, conf_matrix)
print("Hessian: \n", hessian)
show_matrix(hessian, "Hessian matrix")
show_matrix(get_cov(hessian), "Covariance matrix")
# %%
print('''
Getting Hessian matrix from combined hist
''')
fig, ax = plt.subplots(1, 3, figsize=(25, 5))
hessian_comb = np.zeros((len(conf_matrix), len(conf_matrix)))

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        hessian_comb[i][j] = get_hessian_comb(i,j, comb_hist, all_mu)
print("Hessian from comb hist: \n", hessian_comb)
show_matrix(hessian_comb, "Hessian matrix from comb hist", ax[0])
show_matrix(get_cov(hessian_comb), "Covariance matrix from comb hist", ax[1])
correlation_matrix = get_correlation_matrix(get_cov(hessian_comb))
show_matrix(correlation_matrix, "Correlation matrix", ax[2])
print(get_uncertainties(get_cov(hessian_comb)))
# %%
