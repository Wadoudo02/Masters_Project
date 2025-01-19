#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:49:47 2025

@author: wadoudcharbak
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim

# Plotting style
plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300

plot_fraction = True

Quadratic = True

cg = 0.3
ctg = 0.69



features = ["deltaR", "HT", "n_jets", "delta_phi_gg"] 
features = [f"{feature}_sel" for feature in features]



# SMEFT weighting function
def add_SMEFT_weights(proc_data, cg, ctg, name="new_weights", quadratic=False):
    proc_data[name] = proc_data['plot_weight'] * (1 + proc_data['a_cg'] * cg + proc_data['a_ctgre'] * ctg)
    if quadratic:
        proc_data[name] += (
            (cg ** 2) * proc_data["b_cg_cg"]
            + cg * ctg * proc_data["b_cg_ctgre"]
            + (ctg ** 2) * proc_data["b_ctgre_ctgre"]
        )
    return proc_data

# Load and preprocess ttH data
print(" --> Loading process: ttH")
df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
df_tth = df_tth[(df_tth["mass_sel"] == df_tth["mass_sel"])]  # Remove NaNs in the selected variable
df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
df_tth['pt_sel'] = df_tth['pt-over-mass_sel'] * df_tth['mass_sel']


invalid_weights = df_tth["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]

# Split the dataset into two random halves
df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=35)

df_smeft = add_SMEFT_weights(df_smeft, cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

'''
# Add SMEFT weights for classification
df_smeft = add_SMEFT_weights(df_tth.copy(), cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)
df_sm = df_tth.copy()  # SM is treated as the baseline with cg=0, ctg=0
'''

# Normalize the "plot_weight" for df_smeft
df_smeft["plot_weight"] /= df_smeft["plot_weight"].sum()

# Normalize the "plot_weight" for df_sm
df_sm["plot_weight"] /= df_sm["plot_weight"].sum()


df_smeft["plot_weight"] *= 10**4
df_sm["plot_weight"] *= 10**4

# Label SMEFT and SM data
df_smeft['label'] = 1  # SMEFT
df_sm['label'] = 0     # SM

# Combine datasets
df_combined = pd.concat([df_smeft, df_sm])

# Plot input feature distributions for train/test representativeness
print(" --> Plotting input feature distributions...")
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_combined,
        x=feature,
        hue="label",
        weights="plot_weight",
        bins=50,
        element="step",
        common_norm=False,
        kde=False,
        palette={0: "green", 1: "blue"}
    )
    plt.title(f"Feature: {feature} - Train/Test Splits")
    plt.xlabel(feature)
    plt.ylabel("Weighted Count")
    plt.legend(["SM", "SMEFT"])
    plt.show()


# Features and labels
X = df_combined[features]  
y = df_combined['label']                # Target labels (SMEFT vs SM)
weights = df_combined["plot_weight"]


# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
weights_tensor = torch.tensor(weights.values, dtype=torch.float32)

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_tensor, y_tensor, weights_tensor, test_size=0.3, random_state=43
)

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Model, loss function, and optimiser
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
criterion = nn.BCELoss(reduction='none')  # Weighted BCE loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    weighted_loss = (loss * w_train).mean()

    # Backward pass and optimisation
    weighted_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {weighted_loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_proba_test = model(X_test).squeeze()
    y_proba_train = model(X_train).squeeze()

# Compute ROC and AUC for test and train
fpr_test, tpr_test, _ = roc_curve(y_test.numpy(), y_proba_test.numpy(), sample_weight=w_test.numpy())
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train.numpy(), y_proba_train.numpy(), sample_weight=w_train.numpy())
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Test ROC AUC: {roc_auc_test:.4f}")
print(f"Train ROC AUC: {roc_auc_train:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr_test, tpr_test, color="blue", lw=2, label=f"Test ROC (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid()
plt.show()

# Plot Histograms
plt.figure(figsize=(12, 8), dpi=300)

plt.hist(y_proba_test[y_test == 1], bins=50, range=(0, 1),  density=plot_fraction, weights = w_test[y_test == 1], histtype='step', linewidth=2, label=f"SMEFT $(c_g, c_{{tg}}) = ({cg}, {ctg})$")
plt.hist(y_proba_test[y_test == 0], bins=50, range=(0, 1),  density=plot_fraction, histtype='step', weights = w_test[y_test == 0], linewidth=2, label="SM $(c_g, c_{{tg}}) = (0, 0)$")
plt.xlabel("XGBoost Classifier Output")
plt.ylabel("Fraction of Events" if plot_fraction else "Events")

plt.legend(loc = "best")
hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())
plt.tight_layout()
plt.show()



# Generate hard predictions
y_test_pred = (y_proba_test >= 0.5).numpy().astype(int)
y_train_pred = (y_proba_train >= 0.5).numpy().astype(int)

# Compute confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred, sample_weight=w_train)
cm_test = confusion_matrix(y_test, y_test_pred, sample_weight=w_test)

# Plot confusion matrices
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["SM", "SMEFT"])
disp_test.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix - Test Data")
plt.show()

'''
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["SM", "SMEFT"])
disp_train.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix - Train Data")
plt.show()
'''