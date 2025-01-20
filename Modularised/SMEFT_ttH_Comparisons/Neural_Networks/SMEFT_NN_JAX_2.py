#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:48:22 2025

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
from functools import partial
from utils import *

import time

start_time = time.time()

# Plotting style
plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300

plot_fraction = True

Quadratic = True

cg = 0.3
ctg = 0.69

PlotInputFeatures = False

LossPlotLog = True  # Toggle for log scale

seed_number = 42


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
df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=seed_number)

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


if PlotInputFeatures:
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
X = df_combined[features].values
y = df_combined['label'].values
weights = df_combined['plot_weight'].values

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=seed_number
)

# Set random seeds for reproducibility
key = random.PRNGKey(seed_number)

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim):
        self.params = {
            'W1': random.normal(key, (input_dim, hidden_dim)) * jnp.sqrt(2 / input_dim),
            'b1': jnp.zeros((hidden_dim,)),
            'W2': random.normal(key, (hidden_dim, 1)) * jnp.sqrt(2 / hidden_dim),
            'b2': jnp.zeros((1,))
        }

    def forward(self, params, X, key=None):
        Z1 = jnp.dot(X, params['W1']) + params['b1']
        A1 = jax.nn.relu(Z1)
        Z2 = jnp.dot(A1, params['W2']) + params['b2']
        return jax.nn.sigmoid(Z2).squeeze()

# Define the weighted binary cross-entropy loss
@jit
def weighted_bce_loss(params, X, y, weights, key):
    preds = nn.forward(params, X, key)
    bce_loss = - (y * jnp.log(preds + 1e-7) + (1 - y) * jnp.log(1 - preds + 1e-7))
    return jnp.mean(bce_loss * weights)

# Define the optimiser
optimiser = optax.adam(learning_rate)
opt_state = optimiser.init(nn.params)

# Update function
@jit
def update(params, opt_state, X, y, weights, key):
    loss, grads = jax.value_and_grad(weighted_bce_loss)(params, X, y, weights, key)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training configuration
input_dim = X_train.shape[1]
hidden_dim = input_dim * 2  # Example: double the input dimensions
nn = NeuralNetwork(input_dim, hidden_dim)

epochs = 100
train_losses = []
test_losses = []

params = nn.params  # Initialise parameters

# Training loop
for epoch in range(epochs):
    # Training
    params, opt_state, train_loss = update(params, opt_state, X_train, y_train, w_train, key)
    train_losses.append(train_loss)

    # Testing
    test_loss = weighted_bce_loss(params, X_test, y_test, w_test, key)
    test_losses.append(test_loss)

    # Print diagnostics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

'''
# Check for invalid parameters
nan_check = jax.tree_map(jnp.isnan, params)
if any(jax.tree_leaves(nan_check)):
    raise ValueError("NaNs detected in model parameters!")
'''
# Evaluation
key, eval_key = random.split(key)
y_proba_test = nn.forward(params, X_test, eval_key)
key, eval_key = random.split(key)
y_proba_train = nn.forward(params, X_train, eval_key)

# Convert to numpy for sklearn metrics
y_proba_test = np.array(y_proba_test)
y_proba_train = np.array(y_proba_train)
y_test_np = np.array(y_test)
y_train_np = np.array(y_train)
w_test_np = np.array(w_test)
w_train_np = np.array(w_train)

# Compute ROC and AUC
fpr_test, tpr_test, _ = roc_curve(y_test_np, y_proba_test, sample_weight=w_test_np)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train_np, y_proba_train, sample_weight=w_train_np)
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Test ROC AUC: {roc_auc_test:.4f}")
print(f"Train ROC AUC: {roc_auc_train:.4f}")

# Plot training and test loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
if LossPlotLog:
    plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs. Epochs')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Plot Histograms
plt.figure(figsize=(12, 8), dpi=300)

plt.hist(y_proba_test[y_test == 1], bins=50, range=(0, 1),  density=plot_fraction, weights = w_test[y_test == 1], histtype='step', linewidth=2, label=f"SMEFT $(c_g, c_{{tg}}) = ({cg}, {ctg})$")
plt.hist(y_proba_test[y_test == 0], bins=50, range=(0, 1),  density=plot_fraction, histtype='step', weights = w_test[y_test == 0], linewidth=2, label="SM $(c_g, c_{{tg}}) = (0, 0)$")
plt.xlabel("Neural Network Output")
plt.ylabel("Fraction of Events" if plot_fraction else "Events")

plt.legend(loc = "best")
hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())
plt.tight_layout()
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr_test, tpr_test, color="blue", lw=2, label=f"Test ROC (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid()
plt.show()
