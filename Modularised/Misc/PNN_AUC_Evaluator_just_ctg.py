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
    new_w += proc_data["true_weight"] * (ctg_vals ** 2) * proc_data["b_ctgre_ctgre"]
    
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

features = ["deltaR", "HT", "n_jets", "delta_phi_gg", "pt"]
features = [f"{feature}_sel" for feature in features]

features.append("ctg")

import json

# Specify the filename to read the JSON data from
filename = 'data/NN_AUC_Scores.json'

# Read the JSON data back into a Python dictionary
with open(filename, 'r') as file:
    NN_AUC_Scores = json.load(file)



#%% 6) SCAN OVER c_{tg} (KEEP c_g=0), PLOT AUC

# Define a range of ctg values to test
ctg_values = np.linspace(-3, 3, 31)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

cg_lines = [0]#, +0.3, -0.3, +1, -1]

for cg_val in cg_lines:
    auc_scores = []
    for ctg_val in ctg_values:
        
        # Split the dataset into two halves (for SM and SMEFT testing)
        df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
        
        # Set the initial cg and ctg values in the SMEFT test set
        df_smeft_test["ctg"] = ctg_val
        
        df_sm_test["ctg"] = ctg_val
        
        # Label the datasets (0 for SM, 1 for SMEFT)
        df_sm_test["label"] = 0
        df_smeft_test["label"] = 1
        
        # Apply the SMEFT weights based on the initial cg and ctg values
        #df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        
        df_smeft_test["true_weight"] = add_SMEFT_weights_PNN_ctg(df_smeft_test)
        
        #Normalise Weights
        df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
        df_sm_test["true_weight"] *= 10**4
        
        df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
        df_smeft_test["true_weight"] *= 10**4
        
        
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
    
    
    plt.plot(ctg_values, auc_scores, label='PNN', marker = "o")

# Label the axes and add a title
plt.xlabel('ctg value')
plt.ylabel('AUC score')
plt.title('AUC vs ctg')

plt.plot(NN_AUC_Scores["Ctg Values"], NN_AUC_Scores["NN: AUC vs Ctg"], label="NN AUC Score", marker = "o")

#plt.axvline(x=-0.4, color='grey', linestyle='--', label=r'$\mathrm{AUC_{PNN}} < \mathrm{AUC_{NN}}$')
#plt.plot([], [], linestyle='None', label=r'$\mathrm{c_g}=0,\ \mathrm{c_{tg}}=-0.4$')


# Add a legend to distinguish between the different pairs
plt.legend()
plt.grid()

# Display the plot
plt.show()


