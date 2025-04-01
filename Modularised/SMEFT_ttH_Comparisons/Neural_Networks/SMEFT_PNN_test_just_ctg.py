#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:52:20 2025

@author: wadoudcharbak
This script demonstrates how to implement a parameterised neural network 
to handle different SMEFT Wilson coefficient values (c_g, c_tg) all in 
a single model.
"""

# -------------------------------------------------------------------------
#                         IMPORTS & SETTINGS
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
import torch
import torch.nn as nn
import torch.optim as optim
import json

import copy
# Local utilities
from utils import *

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
    new_w += ((ctg_vals ** 2) * proc_data["b_ctgre_ctgre"]) * proc_data["true_weight"] 
    
    return new_w

from sklearn.preprocessing import StandardScaler

# List of features to be normalised (excluding 'ctg')
features_to_normalise = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt_sel"]

# Initialise the scaler
scaler = StandardScaler()

# Fit the scaler on the selected features and transform them
norm_values = scaler.fit_transform(df_tth[features_to_normalise])

# Create new columns with a '_norm' suffix
for i, feature in enumerate(features_to_normalise):
    df_tth[f"{feature}_norm"] = norm_values[:, i]

# Now your training features could be the new normalised columns plus 'ctg'
training_features = [f"{feat}_norm" for feat in features_to_normalise] + ["ctg"]

print(df_tth.head())


training_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt_sel", "ctg"]

# Define our ctg values
ctg_values = np.linspace(-2, 2, 41)

# Optional: shuffle your dataset so each split is representative
df_shuffled = df_tth.sample(frac=1, random_state=seed_number).reset_index(drop=True)

N_total = len(df_shuffled)
subset_size = N_total // 5  # integer division

df_sm_list = []
df_smeft_list = []

for i, ctg_val in enumerate(ctg_values):
    # Slice out one-fifth of the data
    start_idx = i * subset_size
    # For the last slice, make sure we include all remaining events
    end_idx = (i + 1) * subset_size if i < 4 else N_total
    
    df_part = copy.deepcopy(df_shuffled) #df_shuffled.iloc[start_idx:end_idx].copy() # copy.deepcopy(df_shuffled) #
    
    # Assign this part its ctg value
    df_part["ctg"] = ctg_val
    
    # -------------------
    # SM copy (label = 0)
    # -------------------
    df_part_sm = copy.deepcopy(df_part)
    df_part_sm["label"] = 0
    
    # Normalise to 1e4
    df_part_sm["true_weight"] /= df_part_sm["true_weight"].sum()
    df_part_sm["true_weight"] *= 1e4
    
    df_sm_list.append(df_part_sm)
    
    # -----------------------
    # SMEFT copy (label = 1)
    # -----------------------
    df_part_smeft = copy.deepcopy(df_part)
    df_part_smeft["label"] = 1
    
    # Apply reweighting
    df_part_smeft["true_weight"] = add_SMEFT_weights_PNN_ctg(df_part_smeft)
    
    # Normalise to 1e4
    df_part_smeft["true_weight"] /= df_part_smeft["true_weight"].sum()
    df_part_smeft["true_weight"] *= 1e4
    
    df_smeft_list.append(df_part_smeft)

# Concatenate SM and SMEFT partitions
df_sm = pd.concat(df_sm_list, ignore_index=True)
df_smeft = pd.concat(df_smeft_list, ignore_index=True)

# Optionally combine them into a single DataFrame
df_combined = pd.concat([df_sm, df_smeft], ignore_index=True)
df_combined["original_index"] = np.arange(len(df_combined))



# -------------------------------------------------------------------------
#                 OPTIONAL: PLOT INPUT FEATURE DISTRIBUTIONS
# -------------------------------------------------------------------------
# Original features plus the new SMEFT parameters
# (We add "cg" and "ctg" to the set of features.)
PlotInputFeatures = False

plot_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt_sel"]

if PlotInputFeatures:
    print(" --> Plotting input feature distributions...")
    for feat in plot_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_combined,
            x=feat,
            hue="label",
            weights="true_weight",
            bins=50,
            element="step",
            common_norm=False,
            kde=False,
            palette={0: "green", 1: "blue"}
        )
        plt.title(f"Feature: {feat}")
        plt.xlabel(feat)
        plt.ylabel("Weighted Count")
        plt.legend(["SM", "SMEFT"])
        plt.show()

#%%

# Define the features we actually want to plot (excluding 'ctg' itself)
plot_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt_sel"]

# Unique ctg values used above
ctg_values = np.linspace(-1, 1, 21)

# Create a 5x5 grid (5 rows for ctg values, 5 columns for the chosen features)
fig, axes = plt.subplots(nrows=len(ctg_values), ncols=len(plot_features), figsize=(25, 100))

for i, ctg_val in enumerate(ctg_values):
    # Filter the dataframe for the given ctg value
    df_subset = df_combined[df_combined["ctg"] == ctg_val]
    
    for j, feat in enumerate(plot_features):
        ax = axes[i, j]
        
        # Plot SM vs. SMEFT distributions
        sns.histplot(
            data=df_subset,
            x=feat,
            hue="label",
            weights="true_weight",
            bins=50,
            element="step",
            common_norm=False,
            kde=False,
            palette={0: "green", 1: "blue"},
            ax=ax
        )
        
        # Title for each subplot
        ax.set_title(f"ctg = {ctg_val}, {feat}")
        
        # X and Y labels
        ax.set_xlabel(feat)
        ax.set_ylabel("Weighted Count")
        
        # Fix the legend to show "SM" and "SMEFT"
        handles, labels = ax.get_legend_handles_labels()
        # When hue="label", Seaborn auto-creates labels like "0", "1"
        ax.legend(handles, ["SM", "SMEFT"], loc="best")

plt.tight_layout()
plt.show()

#%%

# Define the features we actually want to plot (excluding 'ctg' itself)
plot_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt_sel"]

# Unique ctg values used above
ctg_values = np.linspace(-2, 2, 41)

# Path to save plots
output_path = "/Users/wadoudcharbak/Downloads/plots_for_animation"
os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist

filenames = []

# Loop over each ctg value, creating a separate figure for each
for ctg_val in ctg_values:
    # Filter the dataframe for the given ctg value
    df_subset = df_combined[df_combined["ctg"] == ctg_val]
    
    # Create a figure with 1 row and columns equal to the length of plot_features
    fig, axes = plt.subplots(nrows=1, ncols=len(plot_features), figsize=(25, 10))
    
    # Add a single title at the top
    fig.suptitle(f"ctg = {ctg_val:.2f}", fontsize=30)
    
    # Ensure axes is iterable even if only one feature
    if len(plot_features) == 1:
        axes = [axes]

    for j, feat in enumerate(plot_features):
        
        feature_label = feat.replace("_sel", "")
        
        if feature_label == "pt":
            num_bins, plot_range, logplot, x_label = [50, (0, 1000), False, "$p_T$ [GeV]"]
        else:
            num_bins, plot_range, logplot, x_label = vars_plotting_dict[feature_label]
        
        ax = axes[j]

        # Plot SM vs. SMEFT distributions
        sns.histplot(
            data=df_subset,
            x=feat,
            hue="label",
            weights="true_weight",
            bins=num_bins,
            element="step",
            common_norm=False,
            kde=False,
            palette={0: "green", 1: "blue"},
            ax=ax
        )
        
        # Remove '_sel' from the feature label for a cleaner x-axis name
        
        # X label
        ax.set_xlabel(x_label)
        
        # Only the leftmost subplot shows the y-axis label
        if j == 0:
            ax.set_ylabel("Weighted Count")
        else:
            ax.set_ylabel("")
        
        # Remove subplot title (so we only rely on the main title)
        ax.set_title("")
        
        # Fix the legend to show "SM" and "SMEFT"
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["SM", "SMEFT"], loc="best")

    plt.tight_layout()
    
    # Save the figure for this ctg value (identical naming as before)
    filename = os.path.join(output_path, f"plots_ctg_{ctg_val:.2f}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    filenames.append(filename)
    
#%%

import os
import imageio
from PIL import Image
import numpy as np

# Define the directory where the plots are saved
output_dir = "/Users/wadoudcharbak/Downloads/plots_for_animation"


# Define the output video filename (saved in the same folder)
output_video = os.path.join(output_dir, "features_evolving.mp4")

# Read images without resizing
images = [imageio.imread(fname) for fname in filenames]

# Create the animation video at 7 fps
imageio.mimwrite(output_video, images, fps=7)

print(f"Animation saved as {output_video}")

#%%

import seaborn as sns


# The features (keys) we want to plot in 5x1 (one row per feature)
plot_features = [
    "deltaR",
    "HT",
    "n_jets",
    "delta_phi_gg",
]

# Unique ctg values
ctg_values = nctg_values = [-0.5, -0.2, 0, 0.2, 0.5]

# Define colors from the husl palette
colors = sns.color_palette("husl", len(ctg_values))

# Create a 5x1 figure
fig, axes = plt.subplots(
    nrows=len(plot_features),
    ncols=1,
    figsize=(8, 25),
    sharey=False
)

for i, feat in enumerate(plot_features):
    ax = axes[i]
    
    # Extract histogram config from your existing vars_plotting_dict
    num_bins, plot_range, logplot, x_label = vars_plotting_dict[feat]
    
    feat += "_sel"
    
    # Loop over each ctg and plot SMEFT distribution
    for j, cval in enumerate(ctg_values):
        # Filter DataFrame for SMEFT events (label=1) at the given ctg
        df_smeft_ctg = df_combined[
            (df_combined["label"] == 1) &
            (df_combined["ctg"] == cval)
        ]
        
        df_smeft_ctg["true_weight"] /= df_smeft_ctg["true_weight"].sum()
        
        # Use numpy to histogram the data
        histvals, bin_edges = np.histogram(
            df_smeft_ctg[feat],
            bins=num_bins,
            range=plot_range,
            weights=df_smeft_ctg["true_weight"]
        )
        
        # Step plot for each ctg
        ax.step(
            bin_edges[:-1],
            histvals,
            where="mid",
            color=colors[j],
            linewidth=2,
            label=f"ctg = {cval}"
        )
    
    # Log scale if specified
    if logplot:
        ax.set_yscale("log")
    
    # Axis labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel("Weighted Count")
    ax.legend(loc="best")

plt.tight_layout()
plt.show()

#%%
# -------------------------------------------------------------------------
#               SPLIT DATA INTO TRAIN & TEST, PREPARE TENSORS
# -------------------------------------------------------------------------

X = df_combined[training_features].values
y = df_combined["label"].values
w = df_combined["true_weight"].values

# We also keep the original index as a separate array
idx = df_combined["original_index"].values

X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
    X, y, w, idx,
    test_size=0.3,
    random_state=seed_number
)
#%%

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)
w_train_t = torch.tensor(w_train, dtype=torch.float32)
w_test_t  = torch.tensor(w_test,  dtype=torch.float32)

# Create a DataLoader for mini-batch training
train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t, w_train_t)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#%%

# -------------------------------------------------------------------------
#                   DEFINE OUR NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)


# -------------------------------------------------------------------------
#            MODEL INITIALISATION, LOSS FUNCTION, OPTIMISER
# -------------------------------------------------------------------------
input_dim = X_train_t.shape[1]        # e.g. 6
hidden_dim = input_dim * 3         # arbitrary choice
model = NeuralNetwork(input_dim, hidden_dim)

criterion = nn.BCELoss(reduction="none")  # We'll apply event weights manually
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# -------------------------------------------------------------------------
#                             TRAINING LOOP
# -------------------------------------------------------------------------
epochs = 100
train_losses = []
test_losses = []

best_loss = float('inf')          # Keep track of the minimum test loss
best_model_state = None           # For storing best model parameters

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y, batch_w in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)          # BCE per event
        weighted_loss = (loss * batch_w).mean()     # apply weights
        weighted_loss.backward()
        
        optimizer.step()
        epoch_loss += weighted_loss.item()

    # Average loss across batches
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_t).squeeze()
        loss_test = criterion(outputs_test, y_test_t)
        weighted_loss_test = (loss_test * w_test_t).mean().item()
    test_losses.append(weighted_loss_test)

    # Adjust learning rate
    scheduler.step()

    # Check if this is the best (lowest) test loss so far
    if weighted_loss_test < best_loss:
        best_loss = weighted_loss_test
        best_model_state = model.state_dict()  # Save the model parameters


    # Print diagnostics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d}/{epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: {weighted_loss_test:.4f}, "
              f"Best Test Loss: {best_loss:.4f}")


# -------------------------------------------------------------------------
#           LOAD/RESTORE THE BEST MODEL AFTER TRAINING (Optional)
# -------------------------------------------------------------------------
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored model state with lowest test loss: {best_loss:.4f}")

#%%
# -------------------------------------------------------------------------
#                    PLOT TRAINING AND TEST LOSSES
# -------------------------------------------------------------------------
plot_fraction = True
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')

if LossPlotLog:
    plt.yscale('log')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs Epoch')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


# -------------------------------------------------------------------------
#               EVALUATE THE MODEL (ROC, AUC, ETC.)
# -------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_proba_test  = model(X_test_t).squeeze()
    y_proba_train = model(X_train_t).squeeze()

fpr_test, tpr_test, _ = roc_curve(
    y_test_t.numpy(), 
    y_proba_test.numpy(), 
    sample_weight=w_test_t.numpy()
)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(
    y_train_t.numpy(), 
    y_proba_train.numpy(), 
    sample_weight=w_train_t.numpy()
)
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Test ROC AUC:  {roc_auc_test:.4f}")
print(f"Train ROC AUC: {roc_auc_train:.4f}")


# -------------------------------------------------------------------------
#               PLOT HISTOGRAMS OF NN OUTPUT (SM vs SMEFT)
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 8), dpi=300)
mask_smeft = (y_test_t == 1)
mask_sm    = (y_test_t == 0)

plt.hist(y_proba_test[mask_smeft], bins=50, range=(0, 1), 
         density=plot_fraction, 
         weights=w_test_t[mask_smeft].numpy(),
         histtype='step', linewidth=2,
         label="SMEFT (any $c_{tg}\\neq 0$)")

plt.hist(y_proba_test[mask_sm], bins=50, range=(0, 1),
         density=plot_fraction, 
         weights=w_test_t[mask_sm].numpy(),
         histtype='step', linewidth=2,
         label="SM $(c_{tg}) = (0, 0)$")

plt.xlabel("Neural Network Output")
plt.ylabel("Fraction of Events" if plot_fraction else "Events")
plt.legend(loc="best")
hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
#                            PLOT ROC CURVE
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2,
         label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr_test, tpr_test, color="blue", lw=2,
         label=f"Test ROC (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Parameterised NN")
plt.legend()
plt.grid()
plt.show()


#%%
# -------------------------------------------------------------------------
#                    SAVE THE MODEL & MISC INFO
# -------------------------------------------------------------------------
model_ckpt = {
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim
}
torch.save(model_ckpt, "data/neural_network_parameterised_just_ctg_new_func.pth")


print(f" --> Saved model to 'data/neural_network_parameterised_just_ctg.pth'")


#%%

# -------------------------------------------------------------------------
#                    LIKELIHOOD RATIO PLOT
# -------------------------------------------------------------------------



def compute_nll(X, c_value, model,  weights=None):
    """
    Compute the weighted Negative Log-Likelihood (NLL) for a dataset of events X
    at a specific c_tg hypothesis, using a parameterised neural network 'model'.
    
    Args:
        X (np.ndarray):      Event features of shape (N, D).
        w (np.ndarray):      Event weights of shape (N,).
        c_value (float):     The current c_tg value we are testing.
        model (object):      A parameterised neural network or function 
                             providing predict(X, c)-> f(x|c).
    
    Returns:
        float: Weighted negative log-likelihood at c_value.
    """

    # 1) Get classifier output f(x_i|c) for each event
    f_vals = model(X).squeeze().detach().numpy()

    # 2) Numerical safety to avoid log(0) or division by zero
    eps = 1e-12
    f_vals = np.clip(f_vals, eps, 1. - eps)

    # 3) Per-event log-likelihood ratio = log( f / (1-f) )
    log_lr = np.log(f_vals) - np.log(1. - f_vals)  # shape (N,)

    # 4) Weighted sum:  - sum_i w_i * log_lr_i
    
    if weights is not None:
        w = np.array(weights, dtype=float)
        nll = -np.sum(w * log_lr)
    else:
        nll = -np.sum( log_lr)

    return nll

def compute_likelihood(X, c_value, model,  weights=None):
    """
    Compute the weighted Negative Log-Likelihood (NLL) for a dataset of events X
    at a specific c_tg hypothesis, using a parameterised neural network 'model'.
    
    Args:
        X (np.ndarray):      Event features of shape (N, D).
        w (np.ndarray):      Event weights of shape (N,).
        c_value (float):     The current c_tg value we are testing.
        model (object):      A parameterised neural network or function 
                             providing predict(X, c)-> f(x|c).
    
    Returns:
        float: Weighted negative log-likelihood at c_value.
    """

    # 1) Get classifier output f(x_i|c) for each event
    f_vals = model(X).squeeze().detach().numpy()

    # 2) Numerical safety to avoid log(0) or division by zero
    eps = 1e-12
    f_vals = np.clip(f_vals, eps, 1. - eps)

    # 3) Per-event likelihood ratio =  f / (1-f) )
    L_event = f_vals / (1.0 - f_vals)

    
    if weights is not None:
        w = np.array(weights, dtype=float)
        
        L_total = np.prod(w * L_event)
    else:
        L_total = np.prod(L_event)

    return L_total


# ------------------------------
# Example: 1D ctg scan in [-3, 3]
# ------------------------------
ctg_values = np.linspace(-3, 3, 31)   # 31 points from -3 to 3
log_likelihoods = []
likelihoods = []

# We'll make a copy of the test features as a NumPy array to modify 'ctg' column
X_test_np = X_test_t.clone().cpu().numpy()   # shape (N, D)
w_test_np = w_test_t.clone().cpu().numpy()

for cval in ctg_values:
    # Copy test features
    X_scanned = np.copy(X_test_np)
    
    #try using the df_tth set instead

    # Here we assume that 'ctg' is the 5th column in your "features"
    # i.e. features = [deltaR_sel, HT_sel, n_jets_sel, delta_phi_gg_sel, ctg, pt_sel]
    # => column index = 4 for 'ctg'
    # If you used a different index for 'ctg', adjust here
    X_scanned[:,5] = cval

    # Compute log-likelihood for this ctg
    logL_cval = compute_nll(X_scanned_t, cval, model,w_test_np )
    log_likelihoods.append(logL_cval)
    
    likelihood_cval = compute_likelihood(X_scanned_t, cval, model,w_test_np )
    likelihoods.append(likelihood_cval)
    
# Plot log-likelihood vs ctg
plt.figure(figsize=(8, 6))
plt.plot(ctg_values, log_likelihoods, marker='o')
plt.xlabel(r"$c_{tg}$")
plt.ylabel("Log Likelihood (Weighted)")
plt.title("1D Scan of Weighted Log Likelihood vs. $c_{tg}$")
plt.grid(True)
plt.show()

# Plot likelihood vs ctg
plt.figure(figsize=(8, 6))
plt.plot(ctg_values, likelihoods, marker='o')
plt.xlabel(r"$c_{tg}$")
plt.ylabel(" Likelihood (Weighted)")
plt.title("1D Scan of Weighted Likelihood vs. $c_{tg}$")
plt.grid(True)
plt.show()
