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

# Define the ctg values to use (change as desired)
ctg_values = [1, 0, -1]
n = len(ctg_values)

# Create an n x n grid of subplots
fig, axes = plt.subplots(n, n, figsize=(12, 12), dpi=300, sharex=True, sharey=True)

# Loop over each weight ctg value (rows) and each evaluation ctg value (columns)
for i, weight_ctg in enumerate(ctg_values):
    # Compute weights for the current row using the weight ctg value
    df_weight = copy.deepcopy(df_tth)
    df_weight["ctg"] = weight_ctg
    df_weight["true_weight"] = add_SMEFT_weights_PNN_ctg(df_weight)
    df_weight["true_weight"] /= df_weight["true_weight"].sum()  # normalise weights

    for j, eval_ctg in enumerate(ctg_values):
        # Prepare a separate dataframe for NN evaluation with the eval ctg value
        df_eval = copy.deepcopy(df_weight)
        df_eval["ctg"] = eval_ctg

        # Prepare the input tensor from the feature columns
        nn_input = torch.tensor(df_eval[features].values, dtype=torch.float32)
        
        # Get NN predictions
        with torch.no_grad():
            probabilities = loaded_model(nn_input).squeeze().numpy()
        
        # Plot the histogram on the appropriate subplot using the weights computed above
        ax = axes[i, j]
        ax.hist(probabilities, bins=50, range=(0, 1), weights=df_weight["true_weight"],
                histtype='step', linewidth=2)
        
        # Optionally add titles/labels to the subplots
        if i == 0:
            ax.set_title(f"PNN Eval ctg: {eval_ctg}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"Weights ctg: {weight_ctg}", fontsize=10)

# Add common x and y labels
#fig.text(0.5, 0.04, 'Probability', ha='center', va='center', fontsize=12)
#fig.text(0.06, 0.5, 'Fraction of Events', ha='center', va='center', rotation='vertical', fontsize=12)

# Add an overall title to the figure
#fig.suptitle("Histogram of NN Predictions: Weights vs PNN Evaluation ctg", fontsize=14, y=0.97)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
plt.show()


#%%


df_tth_like = copy.deepcopy(df_tth)

#df_tth_like["ctg"] = 1

#df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
#df_tth_like["true_weight"] /= df_tth_like["true_weight"].sum()
#df_tth_like["true_weight"] *= 1e4

# Define our ctg values
ctg_range = np.linspace(-3, 3, 100)

# Optional: shuffle your dataset so each split is representative
#df_tth = df_tth.sample(frac=1, random_state=seed_number).reset_index(drop=True)

negative_log_likelihood_ratios = []
likelihood = []

for i, ctg_val in enumerate(ctg_range):
    # Slice out one-fifth of the data

    df_tth_like = copy.deepcopy(df_tth)
    
    # Assign this part its ctg value
    df_tth_like["ctg"] = ctg_val
    
   # df_tth_like["true_weight"] = add_SMEFT_weights_PNN_ctg(df_tth_like)
    
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


# Plot log-likelihood vs ctg
plt.figure(figsize=(8, 6))
plt.plot(ctg_range, negative_log_likelihood_ratios, marker='o')
plt.xlabel(r"$c_{tg}$")
plt.ylabel("Log Likelihood (Weighted)")
plt.title("1D Scan of Weighted Log Likelihood vs. $c_{tg}$")
plt.grid(True)
plt.show()
'''
# Plot likelihood vs ctg
plt.figure(figsize=(8, 6))
plt.plot(ctg_range, likelihood, marker='o')
plt.xlabel(r"$c_{tg}$")
plt.ylabel(" Likelihood (Weighted)")
plt.title("1D Scan of Weighted Likelihood vs. $c_{tg}$")
plt.grid(True)
plt.show()
'''