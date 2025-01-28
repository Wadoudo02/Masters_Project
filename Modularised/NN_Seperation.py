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

# Define the NeuralNetwork class as in the original script
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)

# Load the model checkpoint
checkpoint = torch.load("neural_network.pth")

# Instantiate the model
loaded_model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
loaded_model.eval()


import json

# Load the probability values
with open("proba_values.json", "r") as json_file:
    proba_data = json.load(json_file)

max_proba = proba_data["max_proba"]
min_proba = proba_data["min_proba"]

print(f"Max Probability: {max_proba}, Min Probability: {min_proba}")

proba_range = max_proba - min_proba
category_boundaries = [
    min_proba + i * (proba_range / 4) for i in range(5)  # 5 boundaries for 4 categories
]

# category_boundaries[0] = 0
# category_boundaries[4] = 1

plot_entire_chain = True

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

    # Remove rows with negative plot_weight from DataFrame
    dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]

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
    return A * np.exp(-lambd * (x - 100))



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
        w = np.array(dfs[proc]['plot_weight'])[cat_mask]

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
            w = np.array(dfs[proc]['plot_weight'])[cat_mask]
            

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



NLL_Results = NN_NLL_scans(np.linspace(-1, 1, 1000), quadratic_order)


#%%

import json

# Specify the filename to read the JSON data from
filename = 'chi_squared_results.json'

# Read the JSON data back into a Python dictionary
with open(filename, 'r') as file:
    chi_squared_Results = json.load(file)

#%%




compare_frozen_scans(NLL_Results, chi_squared_Results) 
compare_profile_scans(NLL_Results, chi_squared_Results)

