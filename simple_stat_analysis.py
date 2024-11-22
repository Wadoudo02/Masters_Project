#%%
import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *
from background_dist import *
from nll import *
import json

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

comb_hist = {
    'background': np.array([
        182.82081702, 176.80413301, 170.98545975, 165.35828065, 159.91629358, 
        23.11442279, 23.00059749, 22.88733272, 22.77462572, 22.66247373, 
        7.62507397, 7.5749568, 7.52516903, 7.4757085, 7.42657305, 
        159.83801979, 156.1854996, 152.61644456, 149.12894737, 145.72114432, 
        2.49905835, 2.51170773, 2.52442114, 2.5371989, 2.55004134
    ]),
    'ttH_0': np.array([
        0.588479407, 1.55947081, 3.43222315, 1.09065098, 0.310908472, 
        0.00794613479, 0.0299798693, 0.0553777472, 0.0176453116, 0.00405728034, 
        -2.13918807e-05, -5.99256445e-05, -0.000156925244, -4.43437995e-05, -5.16696147e-06, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]),
    'ttH_1': np.array([
        0.0154903808, 0.041049519, 0.0903454614, 0.0287089043, 0.00818395778, 
        0.539333711, 2.03484519, 3.75869359, 1.1976529, 0.275382701, 
        0.00522329879, 0.0146321658, 0.0383167544, 0.0108275152, 0.00126162744, 
        0.00131031321, 0.00454335077, 0.0087317118, 0.00306175703, 0.000540615021, 
        0.0, 0.0, 0.0, 0.0, 0.0
    ]),
    'ttH_2': np.array([
        0.000201894649, 0.000535020949, 0.00117752207, 0.000374178932, 0.000106666021, 
        0.0145943014, 0.0550626512, 0.101709769, 0.0324083348, 0.00745182076, 
        0.349467143, 0.978971604, 2.5635996, 0.724419751, 0.0844097482, 
        0.00131784575, 0.00456946892, 0.00878190739, 0.00307935801, 0.00054372283, 
        0.0, 0.0, 0.0, 0.0, 0.0
    ]),
    'ttH_3': np.array([
        0.000732579966, 0.00194133738, 0.00427266934, 0.00135771796, 0.000387040422, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0148684075, 0.0416512655, 0.10907075, 0.0308211181, 0.00359129194, 
        0.700762651, 2.42980877, 4.66976709, 1.63744435, 0.289123861, 
        0.00169796184, 0.00577659973, 0.0151301349, 0.00507510743, -9.80953016e-05
    ]),
    'ttH_4': np.array([
        0.000704046813, 0.00186572451, 0.00410625375, 0.00130483639, 0.00037196564, 
        0.000494843492, 0.00186698862, 0.00344863491, 0.00109885723, 0.000252666086, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0144667932, 0.0501618358, 0.0964043314, 0.0338039832, 0.00596877575, 
        0.153745289, 0.523053564, 1.36998777, 0.459535567, -0.00888223168
    ]),
    'ggH': np.array([
        0.23288571, 2.5353613, 4.41319377, 0.34706271, 1.07463477, 
        -0.1555497, 2.31917903, 7.1250048, 4.41924358, -0.55071549, 
        0.0, 2.33300283, 3.77420237, -0.05797646, 0.2020542, 
        0.79001682, 4.1536631, 11.10985672, 4.2121388, 0.16886213, 
        1.12744866, -0.06879719, 5.75953519, 0.8858527, 0.22497163
    ]),
    'VBF': np.array([
        -0.22840382, 0.89222305, 0.90632075, 0.13482125, 0.22081918, 
        0.0, 1.41008532, 0.93113996, 0.81970175, 0.07599229, 
        0.0, -0.02358482, 0.47035143, -0.04192673, 0.0, 
        -0.13461457, 1.23484665, 0.39540575, -0.75678962, -0.05131634, 
        0.0, 0.13093512, 0.4658585, 0.0, 0.0
    ]),
    'VH': np.array([
        0.55585534, 0.56338391, 2.77280101, 0.78260608, 0.19200281, 
        0.32423708, 2.6854984, 3.57705075, 0.95410452, 0.18154009, 
        0.51443547, 1.28835586, 2.53036035, 0.71996919, 0.23016383, 
        0.86270873, 2.37392004, 3.55170604, 0.94415424, 0.27967364, 
        -0.05862386, 0.28587711, 0.994218, 0.49600509, 0.0
    ])
}
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
best_mus = np.ones(len(conf_matrix))
fig, axes = plt.subplots(ncols=5, figsize=(25, 5), dpi = 300, sharex=True)

for i in range(len(init_mu)):
    NLL_vals = []
    for mu in mu_vals:
        init_mu[i] = mu  # Set the i-th \mu to the scan value
        # Calculate the NLL for the current set of \mu values
        NLL_vals.append(calc_NLL_comb(comb_hist, init_mu,signal='ttH'))

    vals = find_crossings((mu_vals, TwoDeltaNLL(NLL_vals)), 1.)
    init_mu[i] = 1
    best_mus[i] = vals[0]#[0]
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
#all_mu = init_mu
print("The optimised values of mu are:", best_mus)

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
fig.savefig(f"{analysis_path}/nll_profiled_fit.png")
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
print("All mus:", best_mus)
hessian = np.zeros((len(conf_matrix), len(conf_matrix)))
#hessian = [[0 for i in range(5)] for j in range(5)]
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        hessian[i][j] = get_hessian(i,j, hists, best_mus, conf_matrix)
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
        hessian_comb[i][j] = get_hessian_comb(i,j, comb_hist, best_mus)
print("Hessian from comb hist: \n", hessian_comb)
show_matrix(hessian_comb, "Hessian matrix from comb hist", ax[0])
show_matrix(get_cov(hessian_comb), "Covariance matrix from comb hist", ax[1])
correlation_matrix = get_correlation_matrix(get_cov(hessian_comb))
show_matrix(correlation_matrix, "Correlation matrix", ax[2])
print(get_uncertainties(get_cov(hessian_comb)))
# %%
'''
Chi squared fit of mu(c)
'''
tth_json_path = "TTH.json"
with open(tth_json_path, "r") as file:
    wilson_data = json.load(file)

c_g_coef = wilson_data["data"]["central"]["a_cg"]
c_tg_coef = wilson_data["data"]["central"]["a_ctgre"]
order = [2, 5,3,4, 6]
c_g_coef_ordered = [c_g_coef[i] for i in order]
c_tg_coef_ordered = [c_tg_coef[i] for i in order]

#print(c_g_coef, c_tg_coef)

def mu_c(c_g, c_tg):
    mus = []
    for i in range(len(c_g_coef_ordered)):
        mus.append(float(1 + c_g*c_g_coef_ordered[i] + c_tg*c_tg_coef_ordered[i]))
    return mus
def get_chi_squared(mu, c_g, c_tg, hessian):
#    print("input ", mu, c_g, c_tg)
    del_mu = mu - mu_c(c_g, c_tg)
    chi2= del_mu.T @ hessian @ del_mu

    return chi2
c_vals = np.linspace(-1, 1, 100)
c_tg=0
chi_squared = []

for c_g in c_vals:
    chi2 = get_chi_squared(best_mus, c_g, c_tg, hessian_comb)
    #print(chi2)
    chi_squared.append(chi2)

plt.figure(figsize=(8, 6))
plt.plot(c_vals, chi_squared, label=f"$\\chi^2(c_g, c_{{tg}} = {c_tg})$")
plt.axhline(2.3, color='red', linestyle='--', label='Threshold')
plt.xlabel(r"Wilson coefficient $c_{g}$")
plt.ylabel(r"$\chi^2(c_{g}, c_{tg})$")
plt.title("$\chi^2$ as a function of Wilson coefficient $c_{g}$")
plt.legend()
plt.grid()
plt.show()


# %%
'''
Chi Squared minimisation
'''
init_guess = 0
c_g, c_tg = 0,0

chi_2_c_g = []
chi_2_c_tg = []

for c_g in c_vals:
    res = minimize(lambda x: get_chi_squared(best_mus, c_g, x, hessian_comb),
                    init_guess, method='Nelder-Mead')
    chi_2_c_g.append(res.fun)

for c_tg in c_vals:
    res = minimize(lambda x: get_chi_squared(best_mus, x, c_tg, hessian_comb),
                    init_guess, method='Nelder-Mead')
    chi_2_c_tg.append(res.fun)

best_c_g = c_vals[np.argmin(chi_2_c_g)].round(2)
best_c_tg = c_vals[np.argmin(chi_2_c_tg)].round(2)

def find_confidence_interval(chi_2, c_vals, min_chi_2, delta_chi_2):
    lower_bound = None
    upper_bound = None
    for i, chi in enumerate(chi_2):
        if chi <= min_chi_2 + delta_chi_2:
            if lower_bound is None:
                lower_bound = c_vals[i]
            upper_bound = c_vals[i]
    return lower_bound, upper_bound

conf_interval_68_c_g = find_confidence_interval(chi_2_c_g, c_vals, vals_c_g, 1)
conf_interval_68_c_tg = find_confidence_interval(chi_2_c_tg, c_vals, vals_c_tg, 1)

print(f"Best fit for c_g: {best_c_g}")
print(f"68% confidence interval for c_g: {conf_interval_68_c_g}")
print(f"Best fit for c_tg: {best_c_tg}")
print(f"68% confidence interval for c_tg: {conf_interval_68_c_tg}")

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(c_vals, chi_2_c_g, label=fr"Best fit: {best_c_g:.2f} $^{{+{conf_interval_68_c_g[1] - best_c_g:.2f}}}_{{-{best_c_g - conf_interval_68_c_g[0]:.2f}}}$")
plt.xlabel('c_g')
plt.ylabel('Chi-squared')
plt.title('Profiled Scan over c_g')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(c_vals, chi_2_c_tg,label=fr"Best fit: {best_c_g:.2f} $^{{+{conf_interval_68_c_g[1] - best_c_g:.2f}}}_{{-{best_c_g - conf_interval_68_c_g[0]:.2f}}}$")
plt.xlabel('c_g')
plt.xlabel('c_tg')
plt.ylabel('Chi-squared')
plt.title('Profiled Scan over c_tg')
plt.legend()

plt.tight_layout()
plt.show()

# %%
