import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.integrate import quad
import json

from utils_W8 import *

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

background_estimates = {}

mass_range = (120, 130) # Beeded here to get BG estimate
mass_bins = 5


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
            
            BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
            bin_estimates = []
            for i in range(len(BG_estimate_bin_edges) - 1):
                integral, _ = quad(exponential_decay, bin_edges[i], bin_edges[i + 1], args=(A, lambd))
                bin_estimates.append(integral)
            
            print(f"Background estimates for category {cat}: {bin_estimates}")
            
            # Store the result
            if cat not in background_estimates:
                background_estimates[cat] = {}
            background_estimates[cat][proc] = bin_estimates
            
        

    ax.set_xlabel(sanitized_var_name)
    ax.set_ylabel("Events")

    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best', ncol=2)

    hep.cms.label(f"category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    ext = f"_cat_{cat}"
    if plot_entire_chain:
        #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
        fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
    plt.show()
    #ax.cla()
    
    

#breakpoint()#%%

#%%

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Confusion Matrix

Normalised = False

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
    if proc in ["Data", "VBF", "VH", "background", "ggH"]:
        continue 
    
    #if proc == "Data":
    #    continue  
    
 # Filter out rows where either category is NaN
    valid_entries = dfs[proc].dropna(subset=['category', 'truth_category', 'plot_weight'])
    
    # Create a weighted 2D histogram for truth vs. reconstructed categories
    confusion_matrix, _, _ = np.histogram2d(
        valid_entries['category'].cat.codes,
        valid_entries['truth_category'].cat.codes,
        bins=[len(labels), len(labels)],
        weights=valid_entries['plot_weight']
    )
    #breakpoint()
    confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True) # plot normalised by reco-pt row i.e. each row should sum to 1.
    
    # Save matrix to dictionary
    confusion_matrices[proc] = confusion_matrix_normalized

    # Apply normalization if the switch is set to True
    if Normalised:
        matrix_to_plot = confusion_matrix_normalized
        fmt = '.2%'  # Display as percentage
        title_suffix = " (Normalised)"
    else:
        matrix_to_plot = confusion_matrix
        fmt = '.2f'  # Display raw counts
        title_suffix = " (Raw Counts)"


    if plot_entire_chain:
        
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
    
        # Annotate each cell based on the format specified by the normalisation switch
        for i in range(len(labels)):
            for j in range(len(labels)):
                cell_value = matrix_to_plot[i, j]
                ax.text(j, i, f'{cell_value:{fmt}}', ha='center', va='center', color='black', fontsize=20)
                
    
    #fig.savefig(f"{plot_path}/Confusion_Matrix_{proc}.png", dpi = 300, bbox_inches="tight")

    plt.show()
    
#%%
'''
# Unfiltered ttH as a comparison

tth_unfiltered = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")

tth_unfiltered['pt_sel'] = tth_unfiltered['pt-over-mass_sel'] * tth_unfiltered['mass_sel']

bins = [0, 60, 120, 200, 300, np.inf]  # Define the boundaries for pt
labels = ['0-60', '60-120', '120-200', '200-300', '>300']  # Labels for each category
tth_unfiltered['category'] = pd.cut(tth_unfiltered['pt_sel'], bins=bins, labels=labels, right=False)

tth_unfiltered['truth_category'] = pd.cut(tth_unfiltered['HTXS_Higgs_pt_sel'], bins=bins, labels=labels, right=False)

valid_entries = tth_unfiltered.dropna(subset=['mass_sel', 'plot_weight'])

# Create a weighted 2D histogram for truth vs. reconstructed categories
confusion_matrix, _, _ = np.histogram2d(
    valid_entries['truth_category'].cat.codes,
    valid_entries['category'].cat.codes,
    bins=[len(labels), len(labels)],
    weights=valid_entries['plot_weight']
)
#breakpoint()
confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) # plot normalised by reco-pt row i.e. each row should sum to 1.

# Save matrix to dictionary
confusion_matrices[proc] = confusion_matrix_normalized

# Apply normalization if the switch is set to True
if Normalised:
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
ax.set_title(f"Confusion Matrix for Unfiltered ttH {title_suffix}")

# Annotate each cell based on the format specified by the normalization switch
for i in range(len(labels)):
    for j in range(len(labels)):
        cell_value = matrix_to_plot[i, j]
        ax.text(j, i, f'{cell_value:{fmt}}', ha='center', va='center', color='black', fontsize=20)

#fig.savefig(f"{plot_path}/Confusion_Matrix_{proc}.png", dpi = 300, bbox_inches="tight")

plt.show()
'''

#%%

# Define signal window parameters
hists = {}

mass_bins = 5
v = 'mass_sel'

# Initialize histogram data for each reconstructed category and process
for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        # Apply mask to categorize events by reconstructed category
        if proc == "background":
            hists[cat][proc] = np.array(background_estimates[cat][proc])
        else:
            cat_mask = dfs[proc]['category'] == cat
            hists[cat][proc] = np.histogram(
                dfs[proc][cat_mask][v], 
                mass_bins, 
                mass_range, 
                weights=dfs[proc][cat_mask]['true_weight']
            )[0]


conf_matrix = confusion_matrices['ttH']

combined_histogram = build_combined_histogram(hists, conf_matrix, signal='ttH')


# Define the objective function for NLL
def objective_function(mus, fixed_mu_index, fixed_mu_value):
    """
    Objective function to compute NLL for given mu values, fixing one mu.
    
    Parameters:
        mus (array-like): Array of 4 mu values to optimize.
        fixed_mu_index (int): Index of the mu being scanned (fixed during optimization).
        fixed_mu_value (float): The fixed value for the scanned mu.
    
    Returns:
        float: NLL value for the given set of mu values.
    """
    full_mus = np.insert(mus, fixed_mu_index, fixed_mu_value)  # Reconstruct full mu array
    return calc_NLL(combined_histogram, full_mus, signal='ttH')

# Configuration
mu_values = np.linspace(-2, 5, 100)  # Range for scanning a single mu
mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]
bounds = [(0, 3) for _ in range(4)]  # Bounds for the other mu parameters

frozen_mus = mus_initial.copy()

# Prepare the plots
if plot_entire_chain:
    fig, axes = plt.subplots(nrows=5, figsize=(8, 30), dpi=300, sharex=True)

# Perform both frozen scan and profile scan
for i in range(5):
    frozen_NLL_vals = []
    profile_NLL_vals = []
    
    for mu in mu_values:
        # Frozen scan: keep other mu values constant
        
        frozen_mus[i] = mu
        frozen_NLL_vals.append(calc_NLL(combined_histogram, frozen_mus, signal='ttH'))
        
        # Profile scan: optimize the other mu values
        initial_guess = [mus_initial[j] for j in range(5) if j != i]
        obj_func = lambda reduced_mus: objective_function(reduced_mus, fixed_mu_index=i, fixed_mu_value=mu)
        result = minimize(obj_func, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        profile_NLL_vals.append(result.fun)
    
    # Convert to 2ΔNLL
    frozen_NLL_vals = TwoDeltaNLL(frozen_NLL_vals)
    profile_NLL_vals = TwoDeltaNLL(profile_NLL_vals)
    
    # Find crossings
    frozen_vals = find_crossings((mu_values, frozen_NLL_vals), 1.)
    profile_vals = find_crossings((mu_values, profile_NLL_vals), 1.)
    frozen_label = add_val_label(frozen_vals)
    profile_label = add_val_label(profile_vals)
    
    # Keep optimal Frozen
    
    frozen_mus[i] = frozen_vals[0][0]
    
    if plot_entire_chain:
    # Plotting each NLL curve on a separate subplot
        ax = axes[i]
        ax.plot(mu_values, frozen_NLL_vals, label=f"Frozen Scan: {frozen_label}", color='blue')
        ax.plot(mu_values, profile_NLL_vals, label=f"Profile Scan: {profile_label}", color='red', linestyle='--')
        ax.axvline(1., label="SM (expected)", color='green', alpha=0.5)
        ax.axhline(1, color='grey', alpha=0.5, ls='--')
        ax.axhline(4, color='grey', alpha=0.5, ls='--')
        ax.set_ylim(0, 8)
        ax.legend(loc='best')
        ax.set_ylabel("q = 2$\\Delta$NLL")
        ax.set_title(f"Optimising $\\mu_{i}$")

if plot_entire_chain:
    # Show the plot
    plt.xlabel("$\\mu$ Value")
    plt.tight_layout()
    plt.show()

#%%


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
    return calc_NLL(combined_histogram, mus, signal='ttH')

# Initial values for the five mu parameters
mus_initial = [1.0, 1.0, 1.0, 1.0, 1.0]

# Define bounds for each mu parameter (e.g., between 0 and 3)
bounds = [(0, 3) for _ in range(5)]

# Perform the optimization
result = minimize(objective_function, mus_initial, bounds=bounds, method='L-BFGS-B')

#print(np.array(result.hess_inv.todense()))

# Extract the optimized mu values
optimized_mus = result.x
print("The optimized values of mu are:", optimized_mus)


plot_heatmaps = False


if plot_heatmaps:
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
                    NLL_grid[idx_i, idx_j] = calc_NLL(combined_histogram, mus, signal='ttH')
            
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
       
    
#%%

labels = ['0-60', '60-120', '120-200', '200-300', '>300'] # define labels again for hists

hessian_matrix = calc_Hessian(combined_histogram, optimized_mus)
print("Hessian Matrix:\n", np.array2string(hessian_matrix, precision=4, separator=' ', suppress_small=True))

try:
    covariant_matrix = np.linalg.inv(hessian_matrix)
    print("\nCovariant Matrix:\n", np.array2string(covariant_matrix, precision=4, separator=' ', suppress_small=True))
except np.linalg.LinAlgError:
    print("Error: Hessian matrix is singular and cannot be inverted. Check if the Hessian is non-singular.")

# Extract uncertainties for each mu parameter
uncertainties = np.sqrt(np.diag(covariant_matrix))
print("\nUncertainties in mu parameters:\n", np.array2string(uncertainties, precision=4, separator=', ', suppress_small=True))


correlation_matrix = covariance_to_correlation(covariant_matrix)
print("\nCorrelation Matrix:\n", np.array2string(correlation_matrix, precision=4, separator=' ', suppress_small=True))


#%%

ttH_coefficients_data = json.load(open("data/TTH.json", 'r'))

'''
NOTES ON INDEX IN JSON FILE

[2] = TTH_PTH_0_60 = 0 to 60
[3] = TTH_PTH_120_200 = 120 to 200
[4] = TTH_PTH_200_300 = 200 to 300
[5] = TTH_PTH_60_120 = 60 to 120
[6] = TTH_PTH_GT300 = >300

or

0-60 = [2]
60-120 = [5]
120-200 = [3]
200-300 = [4]
>300 = [6]


'''

optimized_mus = np.array([1,1,1,1,1])

a_cg = ttH_coefficients_data['data']["central"]["a_cg"]
a_ctgre = ttH_coefficients_data['data']["central"]["a_ctgre"]

b_cg_cg = ttH_coefficients_data['data']["central"]["b_cg_cg"]
b_cg_ctgre = ttH_coefficients_data['data']["central"]["b_cg_ctgre"]
b_ctgre_ctgre = ttH_coefficients_data['data']["central"]["b_ctgre_ctgre"]

# CHANGE TO C_G AND C_TG
def mu_c(cg, ctg, quadratic = True):
    """Theoretical model for mu as a function of Wilson coefficient c."""
    
    
    mu_0 = 1 + cg * a_cg[2] + ctg * a_ctgre[2]
    
    mu_1 = 1 + cg * a_cg[5] + ctg * a_ctgre[5]
    
    mu_2 = 1 + cg * a_cg[3] + ctg * a_ctgre[3]
    
    mu_3 = 1 + cg * a_cg[4] + ctg * a_ctgre[4]
    
    mu_4 = 1 + cg * a_cg[6] + ctg * a_ctgre[6]
    
    if quadratic:
        mu_0 += (cg ** 2) * b_cg_cg[2] + cg * ctg * b_cg_ctgre[2] + (ctg ** 2) * b_ctgre_ctgre[2]
        
        mu_1 += (cg ** 2) * b_cg_cg[5] + cg * ctg * b_cg_ctgre[5] + (ctg ** 2) * b_ctgre_ctgre[5]
        
        mu_2 += (cg ** 2) * b_cg_cg[3] + cg * ctg * b_cg_ctgre[3] + (ctg ** 2) * b_ctgre_ctgre[3]
        
        mu_3 += (cg ** 2) * b_cg_cg[4] + cg * ctg * b_cg_ctgre[4] + (ctg ** 2) * b_ctgre_ctgre[4]
        
        mu_4 += (cg ** 2) * b_cg_cg[6] + cg * ctg * b_cg_ctgre[6] + (ctg ** 2) * b_ctgre_ctgre[6]
        
    
    return np.array([mu_0 , mu_1,  mu_2,  mu_3,  mu_4])



# Define range of c
c_values = np.linspace(-1, 2, 100)  # Adjust range as needed
ctg = 0
# Calculate chi-squared values
chi_squared = []
for cg in c_values:
    delta_mu = optimized_mus - mu_c(cg, ctg)
    chi2 = delta_mu.T @ hessian_matrix @ delta_mu
    chi_squared.append(chi2)

#chi2_vals = find_crossings((c_values, chi_squared), 1.)
#chi2_label = add_val_label(chi2_vals)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(c_values, chi_squared, label=f"$\\chi^2(c_g, c_{{tg}} = {ctg})$")

plt.xlabel(r"Wilson coefficient $c_{g}$")
plt.ylabel(r"$\chi^2(c_{g}, c_{tg})$")
plt.title("$\chi^2$ as a function of Wilson coefficient $c_{g}$")
plt.legend()
plt.grid()
plt.show()



cg = 0
# Calculate chi-squared values
chi_squared = []
for ctg in c_values:
    delta_mu = optimized_mus - mu_c(cg, ctg)
    chi2 = delta_mu.T @ hessian_matrix @ delta_mu
    chi_squared.append(chi2)

#chi2_vals = find_crossings((c_values, chi_squared), 1.)
#chi2_label = add_val_label(chi2_vals)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(c_values, chi_squared, label=f"$\\chi^2(c_g = {cg}, c_{{tg}})$")

plt.xlabel(r"Wilson coefficient $c_{tg}$")
plt.ylabel(r"$\chi^2(c_{g}, c_{tg})$")
plt.title("$\chi^2$ as a function of Wilson coefficient $c_{tg}$")
plt.legend()
plt.grid()
plt.show()

#%%
def chi_squared_func(params):
    cg, ctg = params  # Extract parameters
    delta_mu = optimized_mus - mu_c(cg, ctg)
    chi2 = delta_mu.T @ hessian_matrix @ delta_mu  # Calculate chi-squared
    return chi2


initial_guess = [0.0, 0.0]  # Start at cg=0, ctg=0; adjust as needed


result_chi2 = minimize(chi_squared_func, initial_guess, method='Nelder-Mead')

optimal_cg, optimal_ctg = result_chi2.x
min_chi_squared = result_chi2.fun

print(f"Optimal c_g: {optimal_cg}")
print(f"Optimal c_tg: {optimal_ctg}")
print(f"Minimum chi-squared: {min_chi_squared}")


#%%



cg_values = np.linspace(-2, 2, 100)  # Adjust range as needed
ctg_values = np.linspace(-2, 2, 100)  # Adjust range as needed

# Initialize a 2D grid for chi-squared values
chi_squared_grid = np.zeros((len(cg_values), len(ctg_values)))

# Calculate chi-squared for each combination of c_g and c_tg
for i, cg in enumerate(cg_values):
    for j, ctg in enumerate(ctg_values):
        delta_mu = optimized_mus - mu_c(cg, ctg)
        chi_squared_grid[i, j] = delta_mu.T @ hessian_matrix @ delta_mu

contour_levels = [2.3, 5.99]

plt.figure(figsize=(10, 8))


cg_grid, ctg_grid = np.meshgrid(cg_values, ctg_values)  # Create grid for plotting

contour_plot = plt.contour(cg_grid, ctg_grid, chi_squared_grid.T, levels=contour_levels, colors=['yellow', 'green'], linestyles=['--', '-'])
plt.clabel(contour_plot, fmt={2.3: '68%', 5.99: '95%'}, inline=True, fontsize=20)  # Add labels to the contours


plt.contourf(cg_grid, ctg_grid, chi_squared_grid.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
plt.colorbar(label=r"$\chi^2$")

plt.scatter(optimal_cg, optimal_ctg, color='red', label='Minimum $\chi^2$', zorder=5)

plt.xlabel(r"Wilson coefficient $c_{g}$")
plt.ylabel(r"Wilson coefficient $c_{tg}$")
plt.title(r"Heatmap of $\chi^2(c_g, c_{tg}) [Quad Order]$")
plt.legend(frameon=True, edgecolor='black', loc='best')
plt.grid()
plt.show()
