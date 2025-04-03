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
checkpoint = torch.load("data/neural_network_parameterised_turbo.pth")

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



def add_SMEFT_weights_PNN(proc_data):
    cg_vals  = proc_data["cg"]
    ctg_vals = proc_data["ctg"]
    # baseline:
    new_w = proc_data["true_weight"] * (1.0 + proc_data["a_cg"]*cg_vals + proc_data["a_ctgre"]*ctg_vals)
    # optional quadratic:
    new_w += proc_data["true_weight"] * ((cg_vals**2)*proc_data["b_cg_cg"] + (cg_vals*ctg_vals)*proc_data["b_cg_ctgre"] + (ctg_vals**2)*proc_data["b_ctgre_ctgre"])
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
features.append("cg")

import json

# Specify the filename to read the JSON data from
filename = 'data/NN_AUC_Scores_new_func.json'

# Read the JSON data back into a Python dictionary
with open(filename, 'r') as file:
    NN_AUC_Scores = json.load(file)

#%% 5) SCAN OVER c_g (KEEP c_{tg}=0), PLOT AUC

# Define a range of cg values to test
cg_values = np.linspace(-3, 3, 31)


# Create a new figure for the plot
plt.figure(figsize=(12, 6))

plt.style.use(hep.style.CMS)

ctg_lines = [0] #, +0.69] #, -0.69, +1, -1]

for ctg_val in ctg_lines:
    auc_scores = []
    for cg_val in cg_values:
        
        # Split the dataset into two halves (for SM and SMEFT testing)
        df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
        
        # Set the initial cg and ctg values in the SMEFT test set
        df_smeft_test["cg"]  = cg_val
        df_smeft_test["ctg"] = ctg_val
        
        df_sm_test["cg"]  = cg_val
        df_sm_test["ctg"] = ctg_val
        
        # Label the datasets (0 for SM, 1 for SMEFT)
        df_sm_test["label"] = 0
        df_smeft_test["label"] = 1
        
        # Apply the SMEFT weights based on the initial cg and ctg values
        #df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
    
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
    
    
    plt.plot(cg_values, auc_scores, label=f'PNN $c_{{tg}}$={ctg_val}', marker = "o")

# Label the axes and add a title
plt.xlabel('$c_g$ value')
plt.ylabel('AUC score')

plt.plot(NN_AUC_Scores["Cg Values"], NN_AUC_Scores["NN: AUC vs Cg"], label="NN AUC Score", marker = "o")

#plt.axvline(x=-0.4, color='grey', linestyle='--', label=r'$\mathrm{AUC_{PNN}} > \mathrm{AUC_{NN}}$')
#plt.plot([], [], linestyle='None', label=r'$\mathrm{c_g}=-0.4,\ \mathrm{c_{tg}}=0$')

plt.axvline(x=0.3, color='grey', linestyle='--', label=r'NN trained here')

# Add a legend to distinguish between the different pairs
plt.legend()
plt.grid()

# Display the plot
plt.show()

#%% 6) SCAN OVER c_{tg} (KEEP c_g=0), PLOT AUC

# Define a range of ctg values to test
ctg_values = np.linspace(-3, 3, 31)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

cg_lines = [0] #, +0.3] #, -0.3, +1, -1]


for cg_val in cg_lines:
    auc_scores = []
    for ctg_val in ctg_values:
        
        # Split the dataset into two halves (for SM and SMEFT testing)
        df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
        
        # Set the initial cg and ctg values in the SMEFT test set
        df_smeft_test["cg"]  = cg_val
        df_smeft_test["ctg"] = ctg_val
        
        df_sm_test["cg"]  = cg_val
        df_sm_test["ctg"] = ctg_val
        
        # Label the datasets (0 for SM, 1 for SMEFT)
        df_sm_test["label"] = 0
        df_smeft_test["label"] = 1
        
        # Apply the SMEFT weights based on the initial cg and ctg values
        #df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        
        df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        
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
    
    
    plt.plot(ctg_values, auc_scores, label=f'PNN cg={cg_val}', marker = "o")

# Label the axes and add a title
plt.xlabel(r"$c_{tg}$")
plt.ylabel('AUC score')
plt.title(r'AUC vs $c_{tg}$')

plt.plot(NN_AUC_Scores["Ctg Values"], NN_AUC_Scores["NN: AUC vs Ctg"], label="NN AUC Score", marker = "o")

#plt.axvline(x=-0.4, color='grey', linestyle='--', label=r'$\mathrm{AUC_{PNN}} < \mathrm{AUC_{NN}}$')
#plt.plot([], [], linestyle='None', label=r'$\mathrm{c_g}=0,\ \mathrm{c_{tg}}=-0.4$')

plt.axvline(x=0.69, color='grey', linestyle='--', label=r'NN trained here')

# Add a legend to distinguish between the different pairs
plt.legend()
plt.grid()

# Display the plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep  # if you're using HEP-style plots

# Apply CMS style
plt.style.use(hep.style.CMS)

# Define value ranges
cg_values = np.linspace(-3, 3, 31)
ctg_values = np.linspace(-3, 3, 31)

# Containers for AUC scores
auc_vs_cg = []
auc_vs_ctg = []

# Define scan values
ctg_val_fixed = 0
cg_val_fixed = 0

# Scan AUC vs cg
for cg_val in cg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_sm_test["cg"] = df_smeft_test["cg"] = cg_val
    df_sm_test["ctg"] = df_smeft_test["ctg"] = ctg_val_fixed
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)

    df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
    df_sm_test["true_weight"] *= 10**4
    df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
    df_smeft_test["true_weight"] *= 10**4

    auc_score = compute_auc_for_dataset(df_sm_test, df_smeft_test, loaded_model, feature_cols=features)
    auc_vs_cg.append(auc_score)

# Scan AUC vs ctg
for ctg_val in ctg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_sm_test["cg"] = df_smeft_test["cg"] = cg_val_fixed
    df_sm_test["ctg"] = df_smeft_test["ctg"] = ctg_val
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)

    df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
    df_sm_test["true_weight"] *= 10**4
    df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
    df_smeft_test["true_weight"] *= 10**4

    auc_score = compute_auc_for_dataset(df_sm_test, df_smeft_test, loaded_model, feature_cols=features)
    auc_vs_ctg.append(auc_score)

# --- Plotting ---
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# ---- Subplot 1: AUC vs cg ----
ax0 = plt.subplot(gs[0])
ax0.plot(cg_values, auc_vs_cg, label=f'PNN $c_{{tg}}$={ctg_val_fixed}', marker="o")
ax0.plot(NN_AUC_Scores["Cg Values"], NN_AUC_Scores["NN: AUC vs Cg"], label="NN AUC Score", marker="o")
ax0.axvline(x=0.3, color='grey', linestyle='--', label='NN trained here')
ax0.set_xlabel(r"$c_g$")
ax0.set_ylabel("AUC Score")
ax0.set_title("AUC vs $c_g$")
ax0.legend()
ax0.grid(True)

# ---- Subplot 2: AUC vs ctg ----
ax1 = plt.subplot(gs[1])
ax1.plot(ctg_values, auc_vs_ctg, label=f'PNN $c_g$={cg_val_fixed}', marker="o")
ax1.plot(NN_AUC_Scores["Ctg Values"], NN_AUC_Scores["NN: AUC vs Ctg"], label="NN AUC Score", marker="o")
ax1.axvline(x=0.69, color='grey', linestyle='--', label='NN trained here')
ax1.set_xlabel(r"$c_{tg}$")
ax1.set_ylabel("AUC Score")
ax1.set_title("AUC vs $c_{tg}$")
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt


# Define scan ranges
cg_values = np.linspace(-3, 3, 31)
ctg_values = np.linspace(-3, 3, 31)

# Fixed values for scan
ctg_val_fixed = 0
cg_val_fixed = 0

# Containers for AUC scores
auc_vs_cg = []
auc_vs_ctg = []

# Scan AUC vs cg (ctg fixed)
for cg_val in cg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_sm_test["cg"] = df_smeft_test["cg"] = cg_val
    df_sm_test["ctg"] = df_smeft_test["ctg"] = ctg_val_fixed
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)

    df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
    df_sm_test["true_weight"] *= 10**4
    df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
    df_smeft_test["true_weight"] *= 10**4

    auc_score = compute_auc_for_dataset(df_sm_test, df_smeft_test, loaded_model, feature_cols=features)
    auc_vs_cg.append(auc_score)

# Scan AUC vs ctg (cg fixed)
for ctg_val in ctg_values:
    df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
    df_sm_test["cg"] = df_smeft_test["cg"] = cg_val_fixed
    df_sm_test["ctg"] = df_smeft_test["ctg"] = ctg_val
    df_sm_test["label"] = 0
    df_smeft_test["label"] = 1
    df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)

    df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
    df_sm_test["true_weight"] *= 10**4
    df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
    df_smeft_test["true_weight"] *= 10**4

    auc_score = compute_auc_for_dataset(df_sm_test, df_smeft_test, loaded_model, feature_cols=features)
    auc_vs_ctg.append(auc_score)

import matplotlib.pyplot as plt

# --- Plotting ---
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharey=True)

# ---- Top Plot: AUC vs cg ----
ax1.plot(cg_values, auc_vs_cg, label=fr'PNN ($c_{{tg}}$={ctg_val_fixed})', marker="o", linestyle='-', linewidth=2)
ax1.plot(NN_AUC_Scores["Cg Values"], NN_AUC_Scores["NN: AUC vs Cg"], label="NN", marker="s", linestyle='--', linewidth=2)
ax1.axvline(x=0.3, color='grey', linestyle='--', linewidth=1.5, label='NN trained here')

ax1.set_xlabel(r"$c_{g}$")
ax1.set_ylabel("AUC Score")
ax1.set_title("AUC vs $c_{g}$", pad=10)
ax1.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.7)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# ---- Bottom Plot: AUC vs ctg ----
ax2.plot(ctg_values, auc_vs_ctg, label=fr'PNN ($c_{{g}}$={cg_val_fixed})', marker="o", linestyle='-', linewidth=2)
ax2.plot(NN_AUC_Scores["Ctg Values"], NN_AUC_Scores["NN: AUC vs Ctg"], label="NN", marker="s", linestyle='--', linewidth=2)
ax2.axvline(x=0.69, color='grey', linestyle='--', linewidth=1.5, label='NN trained here')

ax2.set_xlabel(r"$c_{tg}$")
ax2.set_ylabel("AUC Score")
ax2.set_title("AUC vs $c_{tg}$", pad=10)
ax2.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.7)
ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig(thesis_plot_path + "/AUC_variation.pdf")

plt.show()




#%% 7) 2D CONTOUR: AUC vs (c_g, c_{tg})

outwards = 2

cg_range = np.linspace(-outwards, outwards, 100)
ctg_range = np.linspace(-outwards,outwards, 100)
PNN_auc_grid = np.zeros((len(cg_range), len(ctg_range)))

for i, cg_val in enumerate(cg_range):
    for j, ctg_val in enumerate(ctg_range):
        # Split the dataset into two halves (for SM and SMEFT testing)
        df_sm_test, df_smeft_test = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
        
        # Set the initial cg and ctg values in the SMEFT test set
        df_smeft_test["cg"]  = cg_val
        df_smeft_test["ctg"] = ctg_val
        
        df_sm_test["cg"]  = cg_val
        df_sm_test["ctg"] = ctg_val
        
        # Label the datasets (0 for SM, 1 for SMEFT)
        df_sm_test["label"] = 0
        df_smeft_test["label"] = 1
        
        # Apply the SMEFT weights based on the initial cg and ctg values
        #df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        
        df_smeft_test["true_weight"] = add_SMEFT_weights_PNN(df_smeft_test)
        
        #Normalise Weights
        df_sm_test["true_weight"] /= df_sm_test["true_weight"].sum()
        df_sm_test["true_weight"] *= 10**4
        
        df_smeft_test["true_weight"] /= df_smeft_test["true_weight"].sum()
        df_smeft_test["true_weight"] *= 10**4
        
        PNN_auc_grid[i, j] = compute_auc_for_dataset(
            df_sm_test,
            df_smeft_test,
            loaded_model,
            feature_cols=features
        )

# Create mesh for plotting
CG, CTG = np.meshgrid(ctg_range, cg_range)  
# We'll put c_{tg} on the x-axis and c_g on the y-axis.

plt.figure(figsize=(8,6))
cs = plt.contourf(CG, CTG, PNN_auc_grid, levels=20, cmap="viridis")
plt.colorbar(cs, label="AUC Score")
plt.xlabel(r"$c_{g}$")
plt.ylabel(r"$c_{tg}$")
plt.title(r"2D Contour of AUC vs $(c_g, c_{tg})$")
plt.show()

np.save("data/PNN_auc_grid_2.npy", PNN_auc_grid)

#%%

# Load the AUC grids
NN_auc_grid = np.load("data/NN_auc_grid.npy")
PNN_auc_grid = np.load("data/PNN_auc_grid.npy")

# Compute the difference: PNN - NN (you can reverse depending on interpretation)
auc_diff_grid = PNN_auc_grid - NN_auc_grid

# Reconstruct the parameter ranges (assuming they were 100 points from -2 to 2)
cg_range = np.linspace(-outwards, outwards, NN_auc_grid.shape[0])
ctg_range = np.linspace(-outwards, outwards, NN_auc_grid.shape[1])
CG, CTG = np.meshgrid(ctg_range, cg_range)

# Plot the difference grid
plt.figure(figsize=(8, 6))

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


from matplotlib.colors import TwoSlopeNorm

# Compute min and max for colour normalisation
vmin = np.min(auc_diff_grid)
vmax = np.max(auc_diff_grid)

# Use TwoSlopeNorm to map 0 to white
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Plot with the new norm
cs = plt.contourf(CG, CTG, auc_diff_grid, levels=20, cmap="RdBu_r", norm=norm)


#plt.contour(CG, CTG, auc_diff_grid, levels=[0], colors='black', linewidths=1.5, linestyles='dashed')


plt.colorbar(cs)
plt.xlabel(r"$c_{tg}$")
plt.ylabel(r"$c_{g}$")
plt.title(r"Difference in AUC: $\mathrm{PNN} - \mathrm{NN}$", pad=15)


plt.plot([0.69], [0.3], marker='o', color='red', markersize=10, lw=0, label='Training point for NN')

plt.legend(loc='upper center',frameon=True, edgecolor='black', fancybox=True, framealpha=0.65, facecolor='white')

plt.tight_layout()

plt.savefig(thesis_plot_path + "/Delta_AUC_2D.pdf")

plt.show()
