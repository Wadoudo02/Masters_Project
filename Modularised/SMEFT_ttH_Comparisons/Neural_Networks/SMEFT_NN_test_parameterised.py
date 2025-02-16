#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:11:41 2025

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

#df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=seed_number)

N = len(df_tth)  # number of events in baseline
cg_min, cg_max = -0.5, 0.5
ctg_min, ctg_max = -0.5, 1

# Draw random c_g and c_tg for each event
# For instance, uniform distribution:
rnd_cg  = np.random.uniform(low=cg_min,  high=cg_max,  size=N)
rnd_ctg = np.random.uniform(low=ctg_min, high=ctg_max, size=N)


# 3) Add columns for c_g, c_tg
df_smeft = df_tth.copy()

df_smeft["cg"]  = rnd_cg
df_smeft["ctg"] = rnd_ctg
df_smeft["label"] = 1  # "SMEFT"

# 4) Reweight to these random parameter values
#    We'll define a function as in your code:
def add_SMEFT_weights_random(proc_data):
    cg_vals  = proc_data["cg"]
    ctg_vals = proc_data["ctg"]
    # baseline:
    new_w = proc_data["true_weight"] * (1.0 + proc_data["a_cg"]*cg_vals + proc_data["a_ctgre"]*ctg_vals)
    # optional quadratic:
    new_w += (cg_vals**2)*proc_data["b_cg_cg"] + (cg_vals*ctg_vals)*proc_data["b_cg_ctgre"] + (ctg_vals**2)*proc_data["b_ctgre_ctgre"]
    return new_w

df_smeft["true_weight"] = add_SMEFT_weights_random(df_smeft)

# 5) Optionally normalise your SMEFT weights
df_smeft["true_weight"] /= df_smeft["true_weight"].sum()
df_smeft["true_weight"] *= 1e4

# 6) If you want some pure SM events labeled "0" (c_g=0, c_tg=0):
rnd_cg_sm  = np.random.uniform(low=cg_min,  high=cg_max,  size=N)
rnd_ctg_sm = np.random.uniform(low=ctg_min, high=ctg_max, size=N)

df_sm = df_tth.copy()
df_sm["label"] = 0
df_sm["cg"] = rnd_cg_sm
df_sm["ctg"] = rnd_ctg_sm
df_sm["true_weight"] /= df_sm["true_weight"].sum()
df_sm["true_weight"] *= 1e4

# 7) Concatenate
df_combined = pd.concat([df_smeft, df_sm], ignore_index=True)
df_combined["original_index"] = np.arange(len(df_combined))

# -------------------------------------------------------------------------
#                 OPTIONAL: PLOT INPUT FEATURE DISTRIBUTIONS
# -------------------------------------------------------------------------
# Original features plus the new SMEFT parameters
# (We add "cg" and "ctg" to the set of features.)
features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "cg", "ctg"]

if PlotInputFeatures:
    print(" --> Plotting input feature distributions...")
    for feat in features:
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


# -------------------------------------------------------------------------
#               SPLIT DATA INTO TRAIN & TEST, PREPARE TENSORS
# -------------------------------------------------------------------------
X = df_combined[features].values
y = df_combined["label"].values
w = df_combined["true_weight"].values

# We also keep the original index as a separate array
idx = df_combined["original_index"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
    X, y, w, idx,
    test_size=0.3,
    random_state=seed_number
)

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
hidden_dim = input_dim * 4            # arbitrary choice
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
         label="SMEFT (any $c_g,c_{tg}\\neq 0$)")

plt.hist(y_proba_test[mask_sm], bins=50, range=(0, 1),
         density=plot_fraction, 
         weights=w_test_t[mask_sm].numpy(),
         histtype='step', linewidth=2,
         label="SM $(c_g, c_{tg}) = (0, 0)$")

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
torch.save(model_ckpt, "data/neural_network_parameterised.pth")

max_proba = float(y_proba_test.max())
min_proba = float(y_proba_test.min())
proba_data = {"max_proba": max_proba, "min_proba": min_proba}

with open("proba_values_PNN_2.json", "w") as json_file:
    json.dump(proba_data, json_file)

print(f" --> Saved model to 'neural_network_parameterised.pth'")
print(f" --> Probability range: min={min_proba}, max={max_proba}")

