#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:38:58 2025

@author: wadoudcharbak
"""
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
import torch.nn as nn
import torch.optim as optim


# Local utilities
from utils import *


from NN_utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)



# Load the model checkpoint
checkpoint = torch.load("data/neural_network_parameterised.pth")

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
#                         SMEFT WEIGHTING FUNCTION
# -------------------------------------------------------------------------
def add_SMEFT_weights(proc_data, cg_val, ctg_val, name="new_weights", quadratic=False):
    """
    For each row in proc_data, calculates the reweighting factor for the 
    specified c_g and c_tg using linear and (optionally) quadratic terms.
    """
    proc_data[name] = proc_data["true_weight"] * (
        1.0 + proc_data["a_cg"] * cg_val + proc_data["a_ctgre"] * ctg_val
    )
    if quadratic:
        proc_data[name] += (
            (cg_val**2) * proc_data["b_cg_cg"]
            + cg_val * ctg_val * proc_data["b_cg_ctgre"]
            + (ctg_val**2) * proc_data["b_ctgre_ctgre"]
        )
    return proc_data


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

# Drop rows with non-positive weights
invalid_weights = (df_tth["plot_weight"] <= 0)
if invalid_weights.any():
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]

def add_SMEFT_weights_random(proc_data):
    cg_vals  = proc_data["cg"]
    ctg_vals = proc_data["ctg"]
    # baseline:
    new_w = proc_data["true_weight"] * (1.0 + proc_data["a_cg"]*cg_vals + proc_data["a_ctgre"]*ctg_vals)
    # optional quadratic:
    new_w += (cg_vals**2)*proc_data["b_cg_cg"] + (cg_vals*ctg_vals)*proc_data["b_cg_ctgre"] + (ctg_vals**2)*proc_data["b_ctgre_ctgre"]
    return new_w



def compute_auc_for_dataset(df_class0, df_class1, model, feature_cols):
    """
    Combines df_class0(label=0) and df_class1(label=1), 
    runs the model, computes weighted AUC.
    """
    #breakpoint()
    df_combined = pd.concat([df_class0, df_class1], ignore_index=True)
    #breakpoint()
    X_data = torch.tensor(df_combined[feature_cols].values, dtype=torch.float32)
    y_true = df_combined["label"].values
    w_data = df_combined["true_weight"].values
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        y_proba = model(X_data).squeeze().numpy()
    
    # Weighted ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba, sample_weight=w_data)
    auc_val = auc(fpr, tpr)
    return auc_val

features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
features = [f"{feature}_sel" for feature in features]
features.append("cg")
features.append("ctg")

import json

# Specify the filename to read the JSON data from
filename = 'data/NN_AUC_Scores.json'

# Read the JSON data back into a Python dictionary
with open(filename, 'r') as file:
    NN_AUC_Scores = json.load(file)

#%% 5) SCAN OVER c_g (KEEP c_{tg}=0), PLOT AUC

# Define a range of cg values to test
cg_values = np.linspace(-2, 2, 20)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Iterate over the different (cg, ctg) pairs
for cg, ctg in [(0, 0.5), (0.5, -0.5), (0.75, 0.5), (0.5, 0.75), (0.75, 0.75)]:
    
    # Split the dataset into two halves (for SM and SMEFT testing)
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    
    # Set the initial cg and ctg values in the SMEFT test set
    df_smeft_test["cg"]  = cg
    df_smeft_test["ctg"] = ctg
    
    df_sm_test["cg"]  = cg
    df_sm_test["ctg"] = ctg
    
    # Label the datasets (0 for SM, 1 for SMEFT)
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    
    # Apply the SMEFT weights based on the initial cg and ctg values
    df_smeft_test["true_weight"] = add_SMEFT_weights_random(df_smeft_test)
    
    auc_scores = []
    
    # Now vary the cg value (keeping ctg = 0) to compute the AUC for each setting
    for cg_val in cg_values:
        df_smeft_test["cg"]  = cg_val
        df_smeft_test["ctg"] = ctg
        
        # Compute the AUC score using your model and the specified features
        auc_score = compute_auc_for_dataset(
            df_sm_test,
            df_smeft_test,
            loaded_model,
            feature_cols=features
        )
        auc_scores.append(auc_score)
    
    # Plot the AUC scores versus the cg values for the current (cg, ctg) pair.
    # The label indicates the initial values used for weighting.
    plt.plot(cg_values, auc_scores, label=f'Initial cg={cg}, ctg={ctg}', marker = "o")

# Label the axes and add a title
plt.xlabel('cg value')
plt.ylabel('AUC score')
plt.title('AUC vs cg for different (cg, ctg) pairs')


plt.plot(NN_AUC_Scores["Cg Values"], NN_AUC_Scores["NN: AUC vs Cg"], label="NN AUC Score", marker = "o")

# Add a legend to distinguish between the different pairs
plt.legend()
plt.grid()

# Display the plot
plt.show()

#%% 6) SCAN OVER c_{tg} (KEEP c_g=0), PLOT AUC

# Define a range of ctg values to test
ctg_values = np.linspace(-2, 2, 20)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Iterate over the different (cg, ctg) pairs
for cg, ctg in [(0, 0), (0.5, -0.5), (0.75, 0.5), (0.5, 0.75), (0.75, 0.75)]:
    
    # Split the dataset into two halves (for SM and SMEFT testing)
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    
    # Set the initial cg and ctg values in the SMEFT test set
    df_smeft_test["cg"]  = cg
    df_smeft_test["ctg"] = ctg
    
    df_sm_test["cg"]  = cg
    df_sm_test["ctg"] = ctg
    
    # Label the datasets (0 for SM, 1 for SMEFT)
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    
    # Apply the SMEFT weights based on the initial cg and ctg values
    df_smeft_test["true_weight"] = add_SMEFT_weights_random(df_smeft_test)
    
    auc_scores = []
    
    # Now vary the cg value (keeping ctg = 0) to compute the AUC for each setting
    for ctg_val in ctg_values:
        df_smeft_test["cg"]  = 0
        df_smeft_test["ctg"] = ctg_val
        
        # Compute the AUC score using your model and the specified features
        auc_score = compute_auc_for_dataset(
            df_sm_test,
            df_smeft_test,
            loaded_model,
            feature_cols=features
        )
        auc_scores.append(auc_score)
    
    # Plot the AUC scores versus the cg values for the current (cg, ctg) pair.
    # The label indicates the initial values used for weighting.
    plt.plot(ctg_values, auc_scores, label=f'Initial cg={cg}, ctg={ctg}', marker = "o")

# Label the axes and add a title
plt.xlabel('ctg value')
plt.ylabel('AUC score')
plt.title('AUC vs ctg for different (cg, ctg) pairs')

plt.plot(NN_AUC_Scores["Ctg Values"], NN_AUC_Scores["NN: AUC vs Ctg"], label="NN AUC Score", marker = "o")

# Add a legend to distinguish between the different pairs
plt.legend()
plt.grid()

# Display the plot
plt.show()

#%% 7) 2D CONTOUR: AUC vs (c_g, c_{tg})
cg_range = np.linspace(-2, 5, 100)
ctg_range = np.linspace(-10, 2, 100)
auc_grid = np.zeros((len(cg_range), len(ctg_range)))

for i, cg_val in enumerate(cg_range):
    for j, ctg_val in enumerate(ctg_range):
        df_smeft_test = df_sm_test.copy()
        df_smeft_test["cg"]  = cg_val
        df_smeft_test["ctg"] = ctg_val
        df_smeft_test["label"] = 1
        df_smeft_test["true_weight"] = add_SMEFT_weights_random(df_smeft_test)
        auc_grid[i, j] = compute_auc_for_dataset(
            df_test_sm,
            df_smeft_test,
            model,
            feature_cols=features
        )

# Create mesh for plotting
CG, CTG = np.meshgrid(ctg_range, cg_range)  
# We'll put c_{tg} on the x-axis and c_g on the y-axis.

plt.figure(figsize=(8,6))
cs = plt.contourf(CG, CTG, auc_grid, levels=20, cmap="viridis")
plt.colorbar(cs, label="AUC Score")
plt.xlabel(r"$c_{tg}$")
plt.ylabel(r"$c_{g}$")
plt.title(r"2D Contour of AUC vs $(c_g, c_{tg})$")
plt.show()