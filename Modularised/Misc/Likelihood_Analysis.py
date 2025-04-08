#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:16:45 2025

@author: wadoudcharbak
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.integrate import quad
import json


import os
print("Current Working Directory:", os.getcwd())

from utils import *

import torch
import torch.nn as nn
import torch.optim as optim



import copy


from NN_utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)



# Load the model checkpoint
checkpoint = torch.load("data/neural_network_parameterised_just_ctg_new_func.pth")

# Instantiate the model
loaded_model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
loaded_model.eval()
# -------------------------------------------------------------------------
#                         IMPORTS & SETTINGS


# Plotting style
plt.style.use(hep.style.CMS)

# Random seed for reproducibility
seed_number = 45
np.random.seed(seed_number)
torch.manual_seed(seed_number)

# Constants
total_lumi = 7.9804
target_lumi = 300

Quadratic = True


PlotInputFeatures = False
LossPlotLog = True  # Toggle for log scale

sample_path="/Users/wadoudcharbak/Downloads/Pass2"

# -------------------------------------------------------------------------
#               LOAD & PREPARE THE BASELINE (ttH) DATAFRAME
# -------------------------------------------------------------------------
print(" --> Loading process: ttH")
df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")

# Remove rows where 'mass_sel' is NaN
df_tth = df_tth[df_tth["mass_sel"].notna()]

# Rescale original weights from full to target luminosity
df_tth["plot_weight"] *= (target_lumi / total_lumi)

df_tth['true_weight'] = df_tth['plot_weight']/10

# Define a derived variable: 'pt_sel' = (pt-over-mass_sel) * mass_sel
df_tth["pt_sel"] = df_tth["pt-over-mass_sel"] * df_tth["mass_sel"]

yield_weight = df_tth["true_weight"].sum()

invalid_weights = df_tth["true_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]
    
df_tth["true_weight"] /= df_tth["true_weight"].sum()
df_tth["true_weight"] *= yield_weight

# Add variables
# Example: (second-)max-b-tag score
b_tag_scores = np.array(df_tth[['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]


# Add nans back in for plotting tools below
max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
df_tth['max_b_tag_score_sel'] = max_b_tag_score
df_tth['second_max_b_tag_score_sel'] = second_max_b_tag_score

# Apply selection: separate ttH from backgrounds + other H production modes
yield_before_sel = df_tth['true_weight'].sum()


mask = df_tth['n_jets_sel'] >= 0
mask = mask & (df_tth['max_b_tag_score_sel'] > 0.4)
#mask = mask & (df_tth['second_max_b_tag_score_sel'] > 0.4)
#mask = mask & (df_tth['HT_sel'] > 200)

df_tth = df_tth[mask]

features = ["deltaR", "HT", "n_jets", "delta_phi_gg", "pt"]
features = [f"{feature}_sel" for feature in features]

features.append("ctg")

# Example reweighting function for ctg
def add_SMEFT_weights_PNN_ctg(proc_data):
    """
    Reweight events according to the chosen ctg value.
    Assumes 'true_weight', 'a_ctgre', and 'b_ctgre_ctgre' are in proc_data.
    """
    ctg_vals = proc_data["ctg"]
    
    # Baseline + linear term
    new_w = proc_data["true_weight"] * (1.0 + proc_data["a_ctgre"] * ctg_vals)
    
    # Optional quadratic term
    new_w += ((ctg_vals ** 2) * proc_data["b_ctgre_ctgre"]) * proc_data["true_weight"]
    
    return new_w


#%%

# -------------------------------------------------------------------------
#               PLOT HISTOGRAMS OF NN OUTPUT (SM vs SMEFT)
# -------------------------------------------------------------------------

# Plot Histograms
plt.figure(figsize=(12, 8), dpi=300)

for ctg_val in np.arange(-2,3):
    # Slice out one-fifth of the data

    df_tth_plot = copy.deepcopy(df_tth)
    
    # Assign this part its ctg value
    df_tth_plot["ctg"] = ctg_val
    
    df_tth_plot["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_plot)
    
    df_tth_plot["true_weight"] /= df_tth_plot["true_weight"].sum()


    # Prepare the input tensor for the PNN
    nn_input = torch.tensor(df_tth_plot[features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        probabilities = loaded_model(nn_input).squeeze().numpy()
        
     # Plot the histogram for the current ctg value
    plt.hist(probabilities, bins=50, range=(0, 1), weights=df_tth_plot["true_weight"],
             histtype='step', linewidth=2, label=f"f({ctg_val})")

# Format the overall plot
plt.xlabel("Probability")
plt.ylabel("Fraction of Events")
plt.title("Histogram of NN Predictions for different ctg values")


plt.legend(loc = "best")
#hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())

plt.tight_layout()
plt.show()

#%%


# Define the \ctg values to use
ctg_values = [-1, 0, 1]
n = len(ctg_values)

# Create a square grid of subplots
fig, axes = plt.subplots(n, n, figsize=(10, 10), dpi=300, sharex=True, sharey=True)

# Loop over each weight \ctg (rows) and evaluation \ctg (columns)
for i, weight_ctg in enumerate(ctg_values):
    df_weight = copy.deepcopy(df_tth)
    df_weight["ctg"] = weight_ctg
    df_weight["true_weight"] = add_SMEFT_weights_PNN_ctg(df_weight)
    df_weight["true_weight"] /= df_weight["true_weight"].sum()  # Normalise

    for j, eval_ctg in enumerate(ctg_values):
        df_eval = copy.deepcopy(df_weight)
        df_eval["ctg"] = eval_ctg

        nn_input = torch.tensor(df_eval[features].values, dtype=torch.float32)

        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()

        ax = axes[i, j]
        ax.hist(probabilities, bins=50, range=(0, 1), weights=df_weight["true_weight"],
                histtype='step', linewidth=1.8, color='red')

        # Axis titles and labels
        if i == 0:
            ax.set_title(f"PNN Evaluated $c_{{tg}}$: {eval_ctg}", fontsize=14)

        # Tick formatting
        ax.tick_params(axis='both', labelsize=10)
        
    for i, weight_ctg in enumerate(ctg_values):
        fig.text(0.1, 0.75 - i * (0.51 / (n - 1)), f"Weights $c_{{tg}}$: {weight_ctg}", 
                 va='center', ha='center', fontsize=14, rotation='vertical')

# Common labels
fig.text(0.56, 0.04, 'PNN Output Probability', ha='center', va='center', fontsize=20)
fig.text(0.05, 0.5, 'Weighted Event Fraction', ha='center', va='center', rotation='vertical', fontsize=20)

# Overall title
#fig.suptitle("Neural Network Predictions vs Weighting and Evaluation $ctg$ Values", fontsize=16, y=0.95)

plt.tight_layout(rect=[0.08, 0.06, 1, 0.92])
plt.savefig(thesis_plot_path + "/Large_PNN_Scan.pdf")
plt.show()

#%%


df_tth_like = copy.deepcopy(df_tth)

ctg = 0.8

df_tth_like["ctg"] = ctg

df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
df_tth_like["true_weight"] *= 1e4

# Define our ctg values
ctg_range = np.linspace(-3, 3, 100)

# Optional: shuffle your dataset so each split is representative
#df_tth = df_tth.sample(frac=1, random_state=seed_number).reset_index(drop=True)

negative_log_likelihood_ratios = []
likelihood = []

for i, ctg_val in enumerate(ctg_range):
    # Slice out one-fifth of the data

    #df_tth_like = copy.deepcopy(df_tth)
    
    # Assign this part its ctg value
    df_tth_like["ctg"] = ctg_val
    
    #df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
    
    # Normalise to 1e4
    df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
    df_tth_like["true_weight"] *= 1e4


    # Prepare the input tensor for the PNN
    nn_input = torch.tensor(df_tth_like[features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        probabilities = loaded_model(nn_input).squeeze().numpy()
        
    w_likelihood = probabilities / (1-probabilities)
    likelihood.append(np.prod(w_likelihood * df_tth_like["true_weight"]))
    
    log_ratios = np.log(probabilities) - np.log(1 - probabilities)
    log_l_ratios = -1*np.sum(log_ratios* df_tth_like["true_weight"] )  # Use sum instead of np.prod
    negative_log_likelihood_ratios.append(log_l_ratios )


negative_log_likelihood_ratios = TwoDeltaNLL(negative_log_likelihood_ratios)

ctg_vals = find_confidence_interval(negative_log_likelihood_ratios, ctg_range, min(negative_log_likelihood_ratios), 1)
ctg_label = add_val_label(ctg_vals)

scaled_ratios = (negative_log_likelihood_ratios - negative_log_likelihood_ratios.min()) / \
                (negative_log_likelihood_ratios.max() - negative_log_likelihood_ratios.min())

# Plot log-likelihood vs ctg
plt.figure(figsize=(8, 6))
plt.axvline(x=0.8, color='grey', linestyle='--', label='$c_{tg} = 0.8$')
plt.plot(ctg_range, scaled_ratios, marker='o', label = f"W($c_{{tg}}$ = {ctg})")
plt.plot([0.8],[0], "s", label = f"Min {ctg_label}", color = "r")

plt.xlabel(r"$c_{tg}$")
plt.ylabel("2$\\Delta$NLL")
#plt.title("1D Scan of Weighted Log Likelihood vs. $c_{tg}$")
plt.grid(True)
plt.legend(loc="best", frameon=True, fancybox=True, fontsize=20)

plt.tight_layout()

plt.savefig(thesis_plot_path + "/NI_log_likelihood.pdf")

plt.show()


#%%


ctg_values = [0.2, 0.5, 0.8, 1.1, 1.4]  # Add as many ctg values as you'd like

results = {}

plt.figure(figsize=(8, 6))

for ctg in ctg_values:
    df_tth_like = copy.deepcopy(df_tth)
    df_tth_like["ctg"] = ctg
    df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
    df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
    df_tth_like["true_weight"] *= 1e4

    ctg_range = np.linspace(-3, 3, 100)
    negative_log_likelihood_ratios = []
    likelihood = []

    for i, ctg_val in enumerate(ctg_range):
        df_tth_like_scan = copy.deepcopy(df_tth_like)
        df_tth_like_scan["ctg"] = ctg_val

        #df_tth_like_scan["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like_scan)
        df_tth_like_scan["true_weight"] /= df_tth_like_scan["true_weight"].sum()
        df_tth_like_scan["true_weight"] *= 1e4

        nn_input = torch.tensor(df_tth_like_scan[features].values, dtype=torch.float32)
        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()

        w_likelihood = probabilities / (1 - probabilities)
        likelihood.append(np.prod(w_likelihood * df_tth_like_scan["true_weight"]))

        log_ratios = np.log(probabilities) - np.log(1 - probabilities)
        log_l_ratios = -1 * np.sum(log_ratios * df_tth_like_scan["true_weight"])
        negative_log_likelihood_ratios.append(log_l_ratios)

    results[ctg] = {
        "ctg_range": ctg_range,
        "log_likelihood_ratios": negative_log_likelihood_ratios,
        "likelihood": likelihood
    }
    
    negative_log_likelihood_ratios = TwoDeltaNLL(negative_log_likelihood_ratios)
    
    ctg_vals = find_confidence_interval(negative_log_likelihood_ratios, ctg_range, min(negative_log_likelihood_ratios), 1)
    ctg_label = add_val_label(ctg_vals)
    
    scaled_ratios = (negative_log_likelihood_ratios - negative_log_likelihood_ratios.min()) / \
                    (negative_log_likelihood_ratios.max() - negative_log_likelihood_ratios.min())
    
    # Plot log-likelihood vs ctg
    plt.plot(ctg_range, scaled_ratios, marker='o', label = f"W($c_{{tg}}$ = {ctg}) {ctg_label}")


plt.xlabel(r"$c_{tg}$")
plt.ylabel("2$\\Delta$NLL")
plt.grid(True)
plt.legend(loc="best", frameon=True, fancybox=True, fontsize=20)

#plt.savefig(f"{thesis_plot_path}/NI_log_likelihood_ctg_{ctg:.2f}.pdf")
plt.show()

#%%

import imageio
import os

# Create a directory to save plots (optional)
output_dir = "/Users/wadoudcharbak/Downloads/plots_for_animation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ctg_range_anim = np.linspace(-2, 2, 41)

filenames = []

for j, ctg_weight in enumerate(ctg_range_anim):

    df_tth_like = copy.deepcopy(df_tth)
    
    ctg = ctg_weight
    
    df_tth_like["ctg"] = ctg
    
    df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
    df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
    df_tth_like["true_weight"] *= 1e4
    
    # Define our ctg values
    ctg_range = np.linspace(-3, 3, 100)
    
    # Optional: shuffle your dataset so each split is representative
    #df_tth = df_tth.sample(frac=1, random_state=seed_number).reset_index(drop=True)
    
    negative_log_likelihood_ratios = []
    likelihood = []
    
    for i, ctg_val in enumerate(ctg_range):
        # Slice out one-fifth of the data
    
        #df_tth_like = copy.deepcopy(df_tth)
        
        # Assign this part its ctg value
        df_tth_like["ctg"] = ctg_val
        
        #df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
        
        # Normalise to 1e4
        df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
        df_tth_like["true_weight"] *= 1e4
    
    
        # Prepare the input tensor for the PNN
        nn_input = torch.tensor(df_tth_like[features].values, dtype=torch.float32)
        
        # Get NN predictions
        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()
            
        w_likelihood = probabilities / (1-probabilities)
        likelihood.append(np.prod(w_likelihood * df_tth_like["true_weight"]))
        
        log_ratios = np.log(probabilities) - np.log(1 - probabilities)
        log_l_ratios = -1*np.sum(log_ratios* df_tth_like["true_weight"] )  # Use sum instead of np.prod
        negative_log_likelihood_ratios.append(log_l_ratios )
        two_delta_NLL = TwoDeltaNLL(negative_log_likelihood_ratios)
    
    
    # Plot log-likelihood vs ctg
    plt.figure(figsize=(10, 8))
    plt.plot(ctg_range, two_delta_NLL, marker='o', label = f"W(ctg = {ctg:.2f})")
    plt.xlabel(r"$c_{tg}$")
    plt.ylabel("2$\\Delta$NLL (Weighted)")
    plt.title("1D Scan of Weighted 2$\\Delta$NLL vs. $c_{tg}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
        # Save the figure
    filename = os.path.join(output_dir, f"2deltaNLL_plot_{j:02d}.png")
    plt.savefig(filename, dpi = 300)
    plt.close()
    filenames.append(filename)
#%%
# Define the directory where the plots are saved
output_dir = "/Users/wadoudcharbak/Downloads/plots_for_animation"

# Collect the filenames of the saved PNG plots (sorted in order)
filenames = sorted([os.path.join(output_dir, fname) for fname in os.listdir(output_dir) if fname.endswith('.png')])

# Read the images
images = [imageio.imread(fname) for fname in filenames]

# Define the output video filename (saved in the same folder)
output_video = os.path.join(output_dir, "ctg_scan_animation_2deltaNLL.mp4")

# Create the animation video at ~20 fps
imageio.mimwrite(output_video, images, fps=7)

print(f"Animation saved as {output_video}")
