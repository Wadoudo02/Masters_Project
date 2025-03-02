#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:44:53 2025

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


from NN_utils import *


# Load the model checkpoint
NN_checkpoint = torch.load("data/neural_network_yielded.pth")

# Instantiate the model
NN_model = NeuralNetwork(NN_checkpoint["input_dim"], NN_checkpoint["hidden_dim"])

# Load model weights
NN_model.load_state_dict(NN_checkpoint["model_state"])

# Set model to evaluation mode
NN_model.eval()


# Load the model checkpoint
PNN_checkpoint = torch.load("data/neural_network_parameterised_yielded.pth")

# Instantiate the model
PNN_model = NeuralNetwork(PNN_checkpoint["input_dim"], PNN_checkpoint["hidden_dim"])

# Load model weights
PNN_model.load_state_dict(PNN_checkpoint["model_state"])

# Set model to evaluation mode
PNN_model.eval()


# Constants
total_lumi = 7.9804
target_lumi = 300

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH x 10", "mediumorchid"],
    "ggH" : ["ggH x 10", "cornflowerblue"],
    "VBF" : ["VBF x 10", "red"],
    "VH" : ["VH x 10", "orange"],
    #"Data" : ["Data", "green"]
}

plot_size = (12, 6)


Quadratic = True

# Load dataframes



dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")


    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass_sel'] == dfs[proc]['mass_sel'])]

    yield_weight = dfs[proc]["plot_weight"].sum()

    # Remove rows with negative plot_weight from DataFrame
    dfs[proc] = dfs[proc][dfs[proc]['plot_weight'] >= 0]
    
    dfs[proc]["plot_weight"] /= dfs[proc]["plot_weight"].sum()
    dfs[proc]["plot_weight"] *= yield_weight

    # Reweight to target lumi
    dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)

    # Calculate true weight: remove x10 multiplier for signal
    if proc in ['ggH', 'VBF', 'VH', 'ttH']:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']/10
    else:
        dfs[proc]['true_weight'] = dfs[proc]['plot_weight']
    

    # Add variables
    # Example: (second-)max-b-tag score
    b_tag_scores = np.array(dfs[proc][['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    
    
    # Add nans back in for plotting tools below
    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score_sel'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score_sel'] = second_max_b_tag_score
    
    # Apply selection: separate ttH from backgrounds + other H production modes
    yield_before_sel = dfs[proc]['true_weight'].sum()
    
    
    mask = dfs[proc]['n_jets_sel'] >= 0
    mask = mask & (dfs[proc]['max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['second_max_b_tag_score_sel'] > 0.4)
    #mask = mask & (dfs[proc]['HT_sel'] > 200)
    
    dfs[proc] = dfs[proc][mask]
    
    yield_after_sel = dfs[proc]['true_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")

    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

#%%


cg_values = np.linspace(-3, 3, 31)

for i, cg_val in enumerate(cg_values):

    # SM COMPARISON ONLY
    # stays at SM as SMEFT weights cannot be changed if not ttH
    
    proc = "ttH"
    
    cg = cg_val
    ctg = 0
    
    plot_fraction = True
    
    dfs[proc]["cg"]  = 0
    dfs[proc]["ctg"]  = 0
    
     # Extract the features for NN input
    features = ["deltaR", "HT", "n_jets", "delta_phi_gg"]
    features = [f"{feature}_sel" for feature in features]
    
    if not all(feature in dfs[proc].columns for feature in features):
        raise ValueError(f"Missing one or more required features in process {proc}")
    
    # Prepare the input tensor for the NN
    NN_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        NN_probabilities = NN_model(NN_input).squeeze().numpy()
        
    # Add the probabilties as a category
    dfs[proc]["NN_probabilities"] = NN_probabilities
    
    features.append("cg")
    features.append("ctg")
    
    # Prepare the input tensor for the NN
    PNN_network_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        PNN_probabilities = PNN_model(PNN_network_input).squeeze().numpy()
        
    # Add the probabilties as a category
    dfs[proc]["PNN_probabilities_0"] = PNN_probabilities
    
    # ---------- Comparison between different PNN modes
    
    dfs[proc]["cg"]  = cg
    dfs[proc]["ctg"]  = ctg
    
    # Prepare the input tensor for the NN
    PNN_network_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        PNN_probabilities = PNN_model(PNN_network_input).squeeze().numpy()
        
    # Add the probabilties as a category
    dfs[proc]["PNN_probabilities"] = PNN_probabilities
    
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    # Grab the data and weights
    x_NN = dfs[proc]["NN_probabilities"]
    x_PNN_0 = dfs[proc]["PNN_probabilities_0"]
    x_PNN = dfs[proc]["PNN_probabilities"]
    
    w = dfs[proc]["true_weight"]
    
    # Normalise to area=1 if requested and non-zero sum
    if plot_fraction and w.sum() > 0:
        w = w / w.sum()
    
    # Plot the histogram
    ax.hist(
        x_NN,
        bins=50,
        range=(0, 1),
        weights=w,
        histtype='step',
        linewidth=2,
        label= proc + " - NN",
        density=False,  # We handle normalisation ourselves
    )
    
    ax.hist(
        x_PNN,
        bins=50,
        range=(0, 1),
        weights=w,
        histtype='step',
        linewidth=2,
        label= proc + f" - PNN({cg:.2f}, {ctg:.2f})",
        density=False,  # We handle normalisation ourselves
        color = "red"
    )
    
    ax.hist(
        x_PNN_0,
        bins=50,
        range=(0, 1),
        weights=w,
        histtype='step',
        linewidth=2,
        label= proc + " - PNN(0, 0)",
        density=False,  # We handle normalisation ourselves
        color = "green"
    )
    
    # Set axis labels
    ax.set_xlabel("Neural Network Output")
    ax.set_ylabel("Fraction of Events" if plot_fraction else "Events")
    
    # Add legend
    ax.legend(loc="best")
    
    # Add the CMS label
    hep.cms.label(proc + " - SM", com="13.6", lumi=target_lumi, ax=ax)
    
    # Final layout adjustments
    plt.tight_layout()
    
    
    plt.savefig(f"/Users/wadoudcharbak/Downloads/plots_for_animation/SM_frame_{i}", dpi = 300)
    
    
    def add_SMEFT_weights_PNN(proc_data):
        cg_vals  = proc_data["cg"]
        ctg_vals = proc_data["ctg"]
        # baseline:
        new_w = proc_data["true_weight"] * (1.0 + proc_data["a_cg"]*cg_vals + proc_data["a_ctgre"]*ctg_vals)
        # optional quadratic:
        new_w += (cg_vals**2)*proc_data["b_cg_cg"] + (cg_vals*ctg_vals)*proc_data["b_cg_ctgre"] + (ctg_vals**2)*proc_data["b_ctgre_ctgre"]
        return new_w
    
    # df_smeft["true_weight"] = add_SMEFT_weights_PNN(df_smeft) just to see how the function works please ignore
    
    # This is for ttH only ad needs to compare SMEFT and SM as well 
    
    proc = "ttH"
    
    plot_fraction = True
    
    #cg = 0.3
    #ctg = 0.69
    
    dfs[proc]["cg"]  = cg
    dfs[proc]["ctg"]  = ctg
    
    
    dfs[proc] = add_SMEFT_weights(dfs[proc], cg=cg, ctg=ctg, name="SMEFT_NN_weight", quadratic=Quadratic)
    
    
    # Prepare the input tensor for the NN
    PNN_network_input = torch.tensor(dfs[proc][features].values, dtype=torch.float32)
    
    # Get NN predictions
    with torch.no_grad():
        PNN_probabilities = PNN_model(PNN_network_input).squeeze().numpy()
        
    # Add the probabilties as a category
    dfs[proc]["PNN_probabilities"] = PNN_probabilities
    
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    # Grab the data and weights
    x_NN = dfs[proc]["NN_probabilities"]
    x_PNN = dfs[proc]["PNN_probabilities"]
    
    w_SMEFT_NN = dfs[proc]["SMEFT_NN_weight"]
    
    dfs[proc]['SMEFT_PNN_weight'] = add_SMEFT_weights_PNN(dfs[proc])
    w_SMEFT_PNN = dfs[proc]['SMEFT_PNN_weight']
    
    # Normalise to area=1 if requested and non-zero sum
    if plot_fraction and w.sum() > 0:
        w = w / w.sum()
    
    # Plot the histogram
    ax.hist(
        x_NN,
        bins=50,
        range=(0, 1),
        weights=w_SMEFT_NN,
        histtype='step',
        linewidth=2,
        label= proc + " - NN",
        density=False  # We handle normalisation ourselves
    )
    
    ax.hist(
        x_PNN,
        bins=50,
        range=(0, 1),
        weights=w_SMEFT_PNN,
        histtype='step',
        linewidth=2,
        label= proc + f" - PNN({cg:.2f}, {ctg:.2f})",
        density=False,  # We handle normalisation ourselves
        color = "red"
    )
    
    # Set axis labels
    ax.set_xlabel("Neural Network Output")
    ax.set_ylabel("Fraction of Events" if plot_fraction else "Events")
    
    # Add legend
    ax.legend(loc="best")
    
    # Add the CMS label
    hep.cms.label(proc + " SMEFT", com="13.6", lumi=target_lumi, ax=ax)
    
    # Final layout adjustments
    plt.tight_layout()
    plt.savefig(f"/Users/wadoudcharbak/Downloads/plots_for_animation/SMEFT_frame_{i}", dpi = 300)


#%%

# The folder containing your frames
image_folder = "/Users/wadoudcharbak/Downloads/plots_for_animation/"

import os
import imageio


# Number of frames: 0 to 30 => 31 frames
n_frames = 31

# Frame rate (10 fps)
fps = 10

# Build a list of sorted frame file paths
filenames = [f"SMEFT_frame_{i}.png" for i in range(n_frames)]
filepaths = [os.path.join(image_folder, fn) for fn in filenames]

images = []
for fp in filepaths:
    images.append(imageio.imread(fp))

# Save to MP4 using imageio's FFmpeg backend
output_path = os.path.join(image_folder, "output_video.mp4")
imageio.mimwrite(output_path, images, fps=fps)

print(f"Created video: {output_path}")
