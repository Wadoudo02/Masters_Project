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

import torch


from NN_utils import *


# Load the model checkpoint
checkpoint = torch.load("data/neural_network_yielded.pth")

# Instantiate the model
loaded_model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
loaded_model.eval()


import json

# Load the probability values
with open("data/proba_values_yielded.json", "r") as json_file:
    proba_data = json.load(json_file)

max_proba = proba_data["max_proba"]
min_proba = proba_data["min_proba"]

print(f"Max Probability: {max_proba}, Min Probability: {min_proba}")

proba_range = max_proba - min_proba
category_boundaries = [
    min_proba + i * (proba_range / 4) for i in range(5)  # 5 boundaries for 4 categories
]


#category_boundaries = [0, 0.24066145, 0.29167122, 0.33349041, 1] # Background Percentiles

#category_boundaries[0] = 0
#category_boundaries[4] = 1

#category_boundaries = [0.0, 0.2152306770648107, 0.34508433673728617, 0.710416158614033, 1.0]

plot_entire_chain = False

plot_fraction = False

# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    #"ttH_SMEFT" : ["ttH_SMEFT x 10", "green"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF x 10", "red"],
    "VH" : ["VH x 10", "orange"],
    #"Data" : ["Data", "green"]
}

plot_size = (12, 8)


cg = 0.3
ctg = 0.69

Quadratic = True

# Load dataframes


# Labels for the categories
labels = ['NN Cat A', 'NN Cat B', 'NN Cat C', 'NN Cat D'] # Labels for each category


dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    
    if proc == "ttH_SMEFT":
        dfs[proc] = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
    else:
        dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")


    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]

    yield_weight = dfs[proc]["plot_weight"].sum()

    # Remove rows with negative plot_weight from DataFrame
    dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]
    
    dfs[proc]["plot_weight"] /= dfs[proc]["plot_weight"].sum()
    dfs[proc]["plot_weight"] *= yield_weight

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
    
    
    mask = dfs[proc]['n_jets_sel'] >= 0
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

    if proc == "ttH_SMEFT":
        dfs[proc] = add_SMEFT_weights(dfs[proc], cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)


     # Extract the features for NN input
    features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
    features = [f"{feature}_sel" for feature in features]
    
    if not all(feature in dfs[proc].columns for feature in features):
        raise ValueError(f"Missing one or more required features in process {proc}")

    # Prepare the input tensor for the NN
    nn_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)

    # Get NN predictions
    with torch.no_grad():
        probabilities = loaded_model(nn_input).squeeze().numpy()
        
    # Add the probabilties as a category
    dfs[proc]["NN_probabilities"] = probabilities

    # Categorise based on probabilities
    dfs[proc]["category"] = pd.cut(
        probabilities,
        bins=category_boundaries,
        labels=labels,
        include_lowest=True
    )

    # Output a quick summary of category distribution
    print(dfs[proc]["category"].value_counts())



cats_unique = labels.copy()

def exponential_decay(x, A, lambd):
    return A * np.exp(-lambd * (x - 120))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot diphoton mass distribution in each category

background_estimates = {}

mass_range = (120, 130)  # Needed here to get BG estimate
mass_bins = 5

v = "mass"
v_dfs = v + "_sel"
for cat in cats_unique:
    print(f" --> Processing: {v} in category {cat}")
    nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]

    # Loop over procs and add histogram
    for proc in procs.keys():
        label, color = procs[proc]

        cat_mask = dfs[proc]['category'] == cat

        x = np.array(dfs[proc][v_dfs][cat_mask])

        # Event weight
        w = np.array(dfs[proc]['true_weight'])[cat_mask]

        counts, bin_edges = np.histogram(x, bins=nbins, range=xrange, weights=w)

        if proc == "background":
           # breakpoint()
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # Perform curve fitting, ignoring empty bins (where counts == 0)
            non_zero_indices = counts > 0
            popt, pcov = curve_fit(exponential_decay, bin_centers[non_zero_indices], counts[non_zero_indices])
            A, lambd = popt  # Unpack fitted parameters

            # Background estimate
            BG_estimate_bin_edges = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
            bin_estimates = []
            for i in range(len(BG_estimate_bin_edges) - 1):
                integral, _ = quad(exponential_decay, BG_estimate_bin_edges[i], BG_estimate_bin_edges[i + 1], args=(A, lambd))
                bin_estimates.append(integral)

            print(f"Background estimates for category {cat}: {bin_estimates}")

            # Store the result
            if cat not in background_estimates:
                background_estimates[cat] = {}
            background_estimates[cat][proc] = bin_estimates

    # Only plot if plot_entire_chain is True
    if plot_entire_chain:
        fig, ax = plt.subplots(1, 1, figsize=plot_size)

        print(f" --> Plotting: {v} in category {cat}")
        for proc in procs.keys():
            label, color = procs[proc]

            cat_mask = dfs[proc]['category'] == cat

            x = np.array(dfs[proc][v_dfs][cat_mask])

            # Event weight
            w = np.array(dfs[proc]['true_weight'])[cat_mask]
            

            counts, bin_edges, _ = ax.hist(x, nbins, xrange, density = plot_fraction, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

            if proc == "background":
                # Plot the fitted exponential decay curve
                x_fit = np.linspace(xrange[0], xrange[1], 1000)
                y_fit = exponential_decay(x_fit, A, lambd)
                ax.plot(x_fit, y_fit, color="red", linestyle="--",
                        label=f"Exponential Fit\n$A={A:.2f}$, $\\lambda={lambd:.4f}$")

        ax.set_xlabel(sanitized_var_name)
        ax.set_ylabel("Events")

        if is_log_scale:
            ax.set_yscale("log")

        ax.legend(loc='best', ncol=2)

        hep.cms.label(f"category {cat}", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

        plt.tight_layout()
        ext = f"_cat_{cat}"
        fig.savefig(f"{plot_path}/{v}{ext}.png", bbox_inches="tight")
        plt.show()
    
#%%

# Looking into the NN, how its categorising the different features

import seaborn as sns


# Suppose you have this flag somewhere in your code:
plot_fraction = True  # or False, depending on your needs

plt.style.use(hep.style.CMS)

# Define your unique category labels
cats_unique = ["NN Cat A", "NN Cat B", "NN Cat C", "NN Cat D"]

# Create a colour palette from seaborn and map each category to a unique colour
palette = sns.color_palette("hls", n_colors=len(cats_unique))
cat_colours = dict(zip(cats_unique, palette))

features_to_plot = ["deltaR", "HT", "n_jets", "delta_phi_gg"]

# Create a 2x2 figure
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

# Loop over each feature and its corresponding subplot
for i, feat in enumerate(features_to_plot):
    ax = axs[i]
    
    # Unpack the list: number of bins, range, log flag, and label text
    bins, rng, logscale, xlabel = vars_plotting_dict[feat]
    
    feat_to_plot = feat + "_sel"

    # For each category, plot a separate histogram
    for cat in cats_unique:
        # Create a mask for the current category
        cat_mask = (dfs["ttH"]["category"] == cat)
        x = dfs["ttH"][feat_to_plot][cat_mask]
        w = dfs["ttH"]["plot_weight"][cat_mask]

        # If requested, normalise the weights so each category's histogram has area=1
        if plot_fraction and len(w) > 0 and w.sum() != 0:
            w = w / w.sum()

        # Plot the histogram for the current category
        ax.hist(
            x,
            bins=bins,
            range=rng,
            weights=w,
            histtype="step",
            lw=2,
            label=cat,
            color=cat_colours[cat]
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.legend(loc="best")

    # If the logscale flag is True, set the y-axis to a logarithmic scale
    if logscale:
        ax.set_yscale("log")

# Add a CMS label to one of the subplots (or you can add it to all if preferred)
hep.cms.label("ttH Features", data=True, lumi=300, com=13.6, ax=axs[0], lumi_format="{0:.0f}")

plt.tight_layout()
plt.show()
# If you want to save to file:
# fig.savefig("NN_feature_distributions.png", dpi=150, bbox_inches="tight")


#%%
plot_fraction = True

if plot_entire_chain:
    # Create a 5x1 figure. Adjust figsize/dpi to your liking.
    fig, axs = plt.subplots(1, 5, figsize=(50, 10), dpi=300, sharex=True)
    axs = axs.flatten()  # In case of indexing convenience

    # Ensure the subplots do not overlap
    plt.tight_layout(pad=3.0)

    # Plot each process in its own subplot
    for i, proc in enumerate(procs.keys()):
        ax = axs[i]

        # Grab the data and weights
        x = dfs[proc]["NN_probabilities"]
        w = dfs[proc]["true_weight"]

        # Normalise to area=1 for this process if requested and non-zero sum
        if plot_fraction and w.sum() > 0:
            w = w / w.sum()  # Now the area under the histogram will be 1.
        
        # Plot the histogram with your chosen binning
        ax.hist(
            x,
            bins=50,
            range=(0, 1),
            weights=w,
            histtype='step',
            linewidth=2,
            label=f"{proc}",
            density=False  # We handle normalisation ourselves
        )

        # Set axis labels
        ax.set_xlabel("Neural Network Output")
        ax.set_ylabel("Fraction of Events" if plot_fraction else "Events")

        # Add legend
        ax.legend(loc="best")

        # Optionally add the CMS label to each subplot
        hep.cms.label(f"{proc}", com="13.6", lumi=target_lumi, ax=ax)

    # Final layout adjustments
    plt.tight_layout()
    plt.show()
#%%

probabilities = dfs["background"]["NN_probabilities"]
weights = dfs["background"]["true_weight"]

# Calculate weighted percentiles
percentiles = np.array([25, 50, 75, 100])
weighted_percentiles = weighted_quantile(probabilities, percentiles/100, weights)


for p, v in zip(percentiles, weighted_percentiles):
    print(f"{p}th percentile: {v:.3f}")

#%%

combined_hist, hists_by_cat = build_combined_histogram_NN(dfs, procs, cats_unique, background_estimates, mass_var="mass_sel", 
                         weight_var="true_weight", mass_range=(120, 130), mass_bins=5)


plot_combined_histogram(combined_hist, categories=cats_unique, mass_bins=5)
                                                                             

#%%


# Parameters for which we want category-wise averages
params = ["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]

cat_averages = {}


for cat in cats_unique:
    # Slice df for this category
    df_cat = dfs["ttH"][dfs["ttH"]["category"] == cat]
    
    # Store each parameter's weighted mean
    cat_averages[cat] = {}
    for param in params:
        cat_averages[cat][param] = get_weighted_average(df_cat, param, "true_weight")


        
        



#%%

# Define signal window parameters
hists = {}

mass_bins = 5
v = 'mass'
v_dfs = v + "_sel"

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
                dfs[proc][cat_mask][v_dfs], 
                mass_bins, 
                mass_range, 
                weights=dfs[proc][cat_mask]['true_weight']
            )[0]
            




#%%



quadratic_order = True



NLL_Results = NN_NLL_scans(hists, np.linspace(-1, 1, 1000), cat_averages, quadratic_order)

Save_Results_to_JSON(NLL_Results, 'data/standard_NN_results.json')

#%%

import json

# Specify the filename to read the JSON data from
filename = 'data/chi_squared_results.json'

# Read the JSON data back into a Python dictionary
with open(filename, 'r') as file:
    chi_squared_Results = json.load(file)

#%%




compare_frozen_scans(NLL_Results, chi_squared_Results) 
compare_profile_scans(NLL_Results, chi_squared_Results)

