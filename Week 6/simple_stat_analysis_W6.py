import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from scipy.optimize import curve_fit

from utils_W6 import *

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
  

def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 100))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category




v = "mass_sel"
for cat in cats_unique:
    fig, ax = plt.subplots(1,1, figsize=plot_size)
    print(f" --> Plotting: {v} in category {cat}")
    nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
    # Loop over procs and add histogram
    for proc in procs.keys():
        label, color = procs[proc]

        cat_mask = dfs[proc]['category']==cat

        x = np.array(dfs[proc][v][cat_mask])

        # Event weight
        w = np.array(dfs[proc]['plot_weight'])[cat_mask]

        counts, bin_edges, _ = ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)
        
        
        if proc == "background":
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
            
            x_BG = np.arange(120.5,130,1)
            BG_estimate = exponential_decay(x_BG, A, lambd)
            
            print(f"For category {cat}, the estimate for the background count for the bins between 120-130 are {BG_estimate}")
            
        

    ax.set_xlabel(sanitized_var_name)
    ax.set_ylabel("Events")

    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best', ncol=2)

    hep.cms.label(f"category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    ext = f"_cat_{cat}"
    #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
    fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
    plt.show()
    #ax.cla()
    
    

#breakpoint()
#%%
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simple binned likelihood fit to mass histograms in signal window (120,130)
hists = {}
mass_range = (120,130)
mass_bins = 5
v = 'mass_sel'

for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        cat_mask = dfs[proc]['category'] == cat
        hists[cat][proc] = np.histogram(dfs[proc][cat_mask][v], mass_bins, mass_range, weights=dfs[proc][cat_mask]['true_weight'])[0]
#breakpoint()
# Calculate NLL as a function of ttH signal strength (assuming fixed bkg and ggH yields)
NLL_vals = []
mu_vals = np.linspace(0,3,100)
for mu in mu_vals:
    NLL_vals.append(calc_NLL(hists, mu))

#breakpoint()
# Plot NLL curve
vals = find_crossings((mu_vals,TwoDeltaNLL(NLL_vals)),1.)
label = add_val_label(vals)

print(" --> Plotting 2NLL curve")
fig, ax = plt.subplots(figsize=plot_size)
ax.plot(mu_vals, TwoDeltaNLL(NLL_vals), label=label)
ax.axvline(1., label="SM (expected)", color='green', alpha=0.5)
ax.axhline(1, color='grey', alpha=0.5, ls='--')
ax.axhline(4, color='grey', alpha=0.5, ls='--')
ax.set_ylim(0,8)
ax.legend(loc='best')
ax.set_xlabel("$\\mu_{ttH}$")
ax.set_ylabel("q = 2$\\Delta$NLL")

plt.tight_layout()
#fig.savefig(f"{plot_path}/2nll_vs_mu.pdf", bbox_inches="tight")
fig.savefig(f"{plot_path}/2nll_vs_mu.png", bbox_inches="tight")
plt.show()
'''
#%%

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Confusion Matrix

Normalised = True

# Define bins and labels for pt categories
bins = [0, 60, 120, 200, 300, np.inf]
labels = ['0-60', '60-120', '120-200', '200-300', '>300']

# Categorise 'HTXS_Higgs_pt_sel' into the same pt categories
for proc in procs.keys():
    if proc == "Data":
        continue  

    dfs[proc]['truth_category'] = pd.cut(dfs[proc]['HTXS_Higgs_pt_sel'], bins=bins, labels=labels, right=False)

# Create confusion matrices for each process
confusion_matrices = {}
for proc in procs.keys():
    if proc in ["Data", "VBF", "VH", "background"]:
        continue 
    
    #if proc == "Data":
    #    continue  
    
 # Filter out rows where either category is NaN
    valid_entries = dfs[proc].dropna(subset=['category', 'truth_category', 'plot_weight'])
    
    # Create a weighted 2D histogram for truth vs. reconstructed categories
    confusion_matrix, _, _ = np.histogram2d(
        valid_entries['truth_category'].cat.codes,
        valid_entries['category'].cat.codes,
        bins=[len(labels), len(labels)],
        weights=valid_entries['plot_weight']
    )
    
    # Save matrix to dictionary
    confusion_matrices[proc] = confusion_matrix
    
    
    # Apply normalization if the switch is set to True
    if Normalised:
        confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True)
        matrix_to_plot = confusion_matrix_normalized
        fmt = '.2%'  # Display as percentage
        title_suffix = " (Normalised)"
    else:
        matrix_to_plot = confusion_matrix
        fmt = '.2f'  # Display raw counts
        title_suffix = " (Raw Counts)"

    

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size for larger plot
    cax = ax.matshow(matrix_to_plot, cmap='Oranges')
    plt.colorbar(cax)
    
    # Set axis labels and title
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Truth pt Category")
    ax.set_ylabel("Reconstructed pt Category")
    ax.set_title(f"Confusion Matrix for {proc}{title_suffix}")

    # Annotate each cell based on the format specified by the normalization switch
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell_value = matrix_to_plot[i, j]
            ax.text(j, i, f'{cell_value:{fmt}}', ha='center', va='center', color='black', fontsize=20)

    #fig.savefig(f"{plot_path}/Confusion_Matrix_{proc}.png", dpi = 300, bbox_inches="tight")

    plt.show()
    
    
#%%

# Define signal window parameters
hists = {}
mass_range = (120, 130)
mass_bins = 5
v = 'mass_sel'

# Initialize histogram data for each reconstructed category and process
for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        # Apply mask to categorize events by reconstructed category
        cat_mask = dfs[proc]['category'] == cat
        hists[cat][proc] = np.histogram(
            dfs[proc][cat_mask][v], 
            mass_bins, 
            mass_range, 
            weights=dfs[proc][cat_mask]['true_weight']
        )[0]


conf_matrix = confusion_matrices['ttH']

# Calculate NLL as a function of the five \mu values for ttH signal strength modifiers
NLL_vals = []
mu_values = np.linspace(0, 3, 100)  # Example range for mu scan

# Define initial \mu values for each of the five truth categories (all start at 1)
mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]
frozen_scan_mus = mus_initial.copy()


#breakpoint()

for i in range(len(frozen_scan_mus)):
    NLL_vals = []
    for mu in mu_values:
        # Adjust the first mu value in the array, keeping others fixed at their initial values
        
        frozen_scan_mus[i] = mu  
    
        # Calculate NLL using the updated calc_NLL function with five \mu values
        NLL_vals.append(calc_NLL(hists, frozen_scan_mus, conf_matrix, signal='ttH', mass_bins=mass_bins))
    
    #breakpoint()
    # Plot the NLL curve as a function of mu
    vals = find_crossings((mu_values, TwoDeltaNLL(NLL_vals)), 1.)
    frozen_scan_mus[i] = vals[0][0]
    label = add_val_label(vals)

    print(" --> Plotting 2NLL curve")
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(mu_values, TwoDeltaNLL(NLL_vals), label=label)
    ax.axvline(1., label="SM (expected)", color='green', alpha=0.5)
    ax.axhline(1, color='grey', alpha=0.5, ls='--')
    ax.axhline(4, color='grey', alpha=0.5, ls='--')
    ax.set_ylim(0, 8)
    ax.legend(loc='best')
    ax.set_xlabel("$\\mu_{ttH}$")
    ax.set_ylabel("q = 2$\\Delta$NLL")
    ax.set_title(f"Optimising $\\mu_{i}$")
    
    plt.tight_layout()
    #fig.savefig(f"{plot_path}/2nll_vs_mu_{i}.png", bbox_inches="tight")
    plt.show()
    
print("The optimised values of mu are:", frozen_scan_mus)

#%%


from scipy.optimize import minimize

# Define the objective function for NLL
def objective_function(mus):
    """
    Objective function to compute NLL for given mu values.
    
    Parameters:
        mus (array-like): Array of 5 mu values for each truth category.
    
    Returns:
        float: NLL value for the given set of mu values.
    """
    # Calculate NLL with the current set of mu values
    return calc_NLL(hists, mus, conf_matrix, signal='ttH', mass_bins=mass_bins)

# Initial values for the five mu parameters
mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]

# Define bounds for each mu parameter (e.g., between 0 and 3)
bounds = [(0, 3) for _ in range(5)]

# Perform the optimization
result = minimize(objective_function, mus_initial, bounds=bounds, method='L-BFGS-B')

# Extract the optimized mu values
optimized_mus = result.x
print("The optimized values of mu are:", optimized_mus)




#%%
'''
# Define ranges for the heatmap
mu_range = np.linspace(0.5, 1.5, 50)

# Create heatmaps for pairs of mu values
for i in range(5):
    for j in range(i+1, 5):
        NLL_grid = np.zeros((len(mu_range), len(mu_range)))
        
        # Fix all mu values to optimized values
        mus = optimized_mus.copy()
        
        # Generate NLL values across the i, j mu grid
        for idx_i, mu_i in enumerate(mu_range):
            for idx_j, mu_j in enumerate(mu_range):
                mus[i] = mu_i  # Vary mu_i
                mus[j] = mu_j  # Vary mu_j
                
                # Calculate NLL with the adjusted mus
                NLL_grid[idx_i, idx_j] = calc_NLL(hists, mus, conf_matrix, signal='ttH', mass_bins=mass_bins)
        
        # Plot the heatmap for this mu pair
        plt.figure(figsize=(8, 6), dpi = 300)
        plt.contourf(mu_range, mu_range, NLL_grid, levels=50, cmap="viridis")
        plt.colorbar(label="NLL")
        plt.xlabel(f"$\\mu_{i+1}$")
        plt.ylabel(f"$\\mu_{j+1}$")
        plt.title(f"NLL Heatmap for $\\mu_{i+1}$ and $\\mu_{j+1}$ with Other $\\mu$ Values Fixed", pad = 20)

        # Add scatter points for both optimized and frozen scan results
        plt.scatter(optimized_mus[i], optimized_mus[j], color='red', label="Simultaneous Optimum", s=100, edgecolor='black')
        plt.scatter(frozen_scan_mus[i], frozen_scan_mus[j], color='blue', label="Frozen Scan Optimum", s=100, edgecolor='black')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
  '''      
#%%


# Assuming `hists`, `mus`, `conf_matrix`, and other parameters are defined
hessian_matrix = calc_Hessian(hists, optimized_mus, conf_matrix, signal='ttH', mass_bins=mass_bins)
print("Hessian Matrix:\n", hessian_matrix)
try:
    covariant_matrix = np.linalg.inv(hessian_matrix)
    print("Covariant Matrix:\n", covariant_matrix)
except np.linalg.LinAlgError:
    print("Error: Hessian matrix is singular and cannot be inverted. Check if the Hessian is non-singular.")
    
    
    
# Extract uncertainties for each mu parameter
uncertainties = np.sqrt(np.diag(covariant_matrix))
print("Uncertainties in mu parameters:", uncertainties)
