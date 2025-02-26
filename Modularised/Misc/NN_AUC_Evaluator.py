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
checkpoint = torch.load("data/neural_network_yielded.pth")

# Instantiate the model
model = NeuralNetwork(checkpoint["input_dim"], checkpoint["hidden_dim"])

# Load model weights
model.load_state_dict(checkpoint["model_state"])

# Set model to evaluation mode
model.eval()
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

yield_weight = df_tth["true_weight"].sum()

invalid_weights = df_tth["true_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]
    
df_tth["true_weight"] /= df_tth["true_weight"].sum()
df_tth["true_weight"] *= yield_weight

features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
features = [f"{feature}_sel" for feature in features]


#%% 4) FUNCTION TO COMPUTE AUC GIVEN TWO DATAFRAMES (SM + pseudo-SMEFT)
def compute_auc_for_dataset(df_class0, df_class1, model, feature_cols):
    """
    Combines df_class0(label=0) and df_class1(label=1), 
    runs the model, computes weighted AUC.
    """
    #breakpoint()
    df_combined = pd.concat([df_class0, df_class1], ignore_index=True)
    
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

#%% 5) SCAN OVER c_g (KEEP c_{tg}=0), PLOT AUC
cg_values = np.linspace(-3, 3, 31)
auc_vs_cg = []


for cg_val in cg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_smeft_test["label"] = 1
    df_sm_test["label"] = 0
    df_smeft_test = add_SMEFT_weights(df_smeft_test, cg_val, 0, name="true_weight", quadratic=Quadratic)
    auc_score = compute_auc_for_dataset(
        df_sm_test,
        df_smeft_test,
        model,
        feature_cols=features
    )
    auc_vs_cg.append(auc_score)

# Plot
plt.figure(figsize=(8,6))
plt.plot(cg_values, auc_vs_cg, marker='o')
plt.xlabel(r"$c_g$")
plt.ylabel("AUC")
plt.title(r"AUC vs $c_g$ (with $c_{tg}=0$)")
plt.grid(True)
plt.show()

#%% 6) SCAN OVER c_{tg} (KEEP c_g=0), PLOT AUC
ctg_values = np.linspace(-3, 3, 31)
auc_vs_ctg = []

for ctg_val in ctg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_smeft_test["label"] = 1
    df_sm_test["label"] = 0
    df_smeft_test = add_SMEFT_weights(df_smeft_test, 0, ctg_val, name="true_weight", quadratic=Quadratic)
    auc_score = compute_auc_for_dataset(
        df_sm_test,
        df_smeft_test,
        model,
        feature_cols=features
    )
    auc_vs_ctg.append(auc_score)


plt.figure(figsize=(8,6))
plt.plot(ctg_values, auc_vs_ctg, marker='s', color='red')
plt.xlabel(r"$c_{tg}$")
plt.ylabel("AUC")
plt.title(r"AUC vs $c_{tg}$ (with $c_g=0$)")
plt.grid(True)
plt.show()

#%%

NN_AUC_Scores = {
    "Cg Values": cg_values,
    "NN: AUC vs Cg": auc_vs_cg,
    "Ctg Values": ctg_values,
    "NN: AUC vs Ctg": auc_vs_ctg,
    }

Save_Results_to_JSON(NN_AUC_Scores, 'data/NN_AUC_Scores.json')

#%% 7) 2D CONTOUR: AUC vs (c_g, c_{tg})
cg_range = np.linspace(-2, 2, 20)
ctg_range = np.linspace(-2, 2, 20)
auc_grid = np.zeros((len(cg_range), len(ctg_range)))

for i, cg_val in enumerate(cg_range):
    for j, ctg_val in enumerate(ctg_range):
        df_smeft_test = df_sm_test.copy()
        df_smeft_test["label"] = 1
        df_smeft_test = add_SMEFT_weights(df_smeft_test, cg_val, ctg_val, name="true_weight", quadratic=Quadratic)
        auc_grid[i, j] = compute_auc_for_dataset(
            df_sm_test,
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

