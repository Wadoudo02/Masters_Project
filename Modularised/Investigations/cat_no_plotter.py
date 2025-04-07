#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:05:51 2025

@author: wadoudcharbak
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import thesis_plot_path

PNN_pos_cg_bounds = np.load("data/PNN_pos_cg_bounds.npy")
PNN_pos_ctg_bounds = np.load("data/PNN_pos_ctg_bounds.npy")

NN_pos_cg_bounds = np.load("data/NN_pos_cg_bounds.npy")
NN_pos_ctg_bounds = np.load("data/NN_pos_ctg_bounds.npy")

cat_no_values = list(range(1, 11))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

font_size = 18

# Plot for cg
ax1.plot(cat_no_values, PNN_pos_cg_bounds, marker='o', linestyle='-', color='tab:blue', label='PNN')
ax1.plot(cat_no_values, NN_pos_cg_bounds, marker='o', linestyle='--', color='tab:orange', label='NN')
ax1.set_ylabel(r'Positive $c_g$ Bound', fontsize=font_size)
ax1.set_title(r'Positive $c_g$ and $c_{tg}$ Bounds vs Number of Categories', fontsize=17)
ax1.grid(True)

legend = ax1.legend(loc="upper right", frameon=True, fancybox=True, fontsize=14)
legend.get_frame().set_edgecolor('black')  # Sets the border colour
legend.get_frame().set_linewidth(1.5)        # Sets the border width
legend.get_frame().set_facecolor('white')    # Optional: sets the background colour

# Plot for ctg
ax2.plot(cat_no_values, PNN_pos_ctg_bounds, marker='o', linestyle='-', color='tab:blue', label='PNN')
ax2.plot(cat_no_values, NN_pos_ctg_bounds, marker='o', linestyle='--', color='tab:orange', label='NN')
ax2.set_ylabel(r'Positive $c_{tg}$ Bound', fontsize=font_size)
ax2.set_xlabel('Number of Categories', fontsize=font_size)
ax2.grid(True)

legend = ax2.legend(loc="upper right", frameon=True, fancybox=True, fontsize=14)
legend.get_frame().set_edgecolor('black')  # Sets the border colour
legend.get_frame().set_linewidth(1.5)        # Sets the border width
legend.get_frame().set_facecolor('white')    # Optional: sets the background colour

# Set x-axis ticks
ax2.set_xticks(cat_no_values)

# Make x and y tick labels smaller for ax1
ax1.tick_params(axis='both', labelsize=14)  # Change 10 to desired font size
# Optional: make tick marks themselves smaller
ax1.tick_params(axis='both', length=4, width=1)  # adjust as needed

# Make x and y tick labels smaller for ax2
ax2.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', length=4, width=1)

# Formatting
plt.tight_layout()
plt.savefig(thesis_plot_path + "/WC_vs_no_of_cats.pdf")
plt.show()