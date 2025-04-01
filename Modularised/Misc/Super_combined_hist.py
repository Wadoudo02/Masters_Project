#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:13:49 2025

@author: wadoudcharbak
"""

from utils import *

combined_hist_STXS = {'background': [22.99458201, 21.43464867, 19.98053991, 18.62507669, 17.36156696,
        30.65319679, 29.2338392 , 27.88020318, 26.58924556, 25.3580641 ,
         6.61564138,  6.55528326,  6.49547581,  6.43621402,  6.37749291,
         2.20391568,  2.19348597,  2.18310562,  2.17277439,  2.16249205,
         1.17692484,  1.18333425,  1.18977857,  1.19625799,  1.20277269],
 'ttH': [0.51498348, 1.59791642, 2.82843305, 0.97497063, 0.25078048,
        0.61654632, 2.46995966, 4.27335104, 1.5885948 , 0.2499453 ,
        0.62449072, 2.13493481, 3.78611632, 1.26434315, 0.23270291,
        0.26528382, 1.00343207, 2.33330162, 0.77439867, 0.09681433,
        0.12708448, 0.57316166, 1.54141894, 0.44179699, 0.02745425],
 'ggH': [0.01219481, 0.33198388, 0.54830694, 0.08485124, 0.15763358,
        0.13464626, 0.59593078, 1.58915206, 0.46015546, 0.07922579,
        0.07740092, 0.50200697, 1.04202877, 0.58055282, 0.08674142,
        0.        , 0.25519268, 0.59104561, 0.20148906, 0.        ,
        0.16379159, 0.13879287, 0.79976236, 0.10468568, 0.03268305],
 'VBF': [0.        , 0.1019075 , 0.10215009, 0.02576572, 0.00882407,
        0.        , 0.09658207, 0.12956475, 0.        , 0.01513989,
        0.        , 0.16362599, 0.13584206, 0.08678179, 0.00711178,
        0.        , 0.        , 0.0505707 , 0.00804697, 0.        ,
        0.        , 0.01225363, 0.05148076, 0.        , 0.        ],
 'VH': [0.07555278, 0.16530854, 0.31992394, 0.11616798, 0.04605256,
        0.14640639, 0.2844566 , 0.52424082, 0.2108679 , 0.05540292,
        0.05579789, 0.35682235, 0.51744536, 0.16041908, 0.01900748,
        0.05304143, 0.16401835, 0.33983795, 0.09675298, 0.04196463,
        0.        , 0.06010375, 0.15149949, 0.04549892, 0.        ]}

cats_unique_STXS = ['0-60', '60-120', '120-200', '200-300', '>300']

combined_hist_NN = {'background': [1592.33151504, 1457.86090769, 1334.74619203, 1222.02837579,
        1118.82945248,  716.66177901,  675.0476141 ,  635.84984529,
         598.92816048,  564.15039506,  618.13496505,  593.1681267 ,
         569.20971379,  546.2189954 ,  524.15688578,  154.39525597,
         151.38460566,  148.43266192,  145.53827998,  142.70033742],
 'ttH': [ 0.60642147,  1.67656538,  2.36806966,  1.01981642,  0.15795786,
         0.92996609,  2.83254307,  4.84189405,  1.73061663,  0.39424954,
         1.92129675,  7.26358152, 12.54206832,  4.64791407,  0.81765812,
         2.67962774, 11.11874085, 22.02850031,  7.64059732,  0.97731003],
 'ggH': [ 9.39273494, 25.21809464, 36.72933503, 15.88161111,  5.34879413,
         5.21368427, 19.00330279, 25.38112579,  9.4760786 ,  2.29286888,
         5.54370651, 19.94154543, 33.01787725, 10.99692449,  2.71429228,
         4.888126  , 18.21392737, 34.56081739, 11.989399  ,  2.83194997],
 'VBF': [0.57306102, 1.65626287, 2.35777325, 1.02664048, 0.30831504,
        0.70236821, 2.09498339, 2.94487235, 1.39081911, 0.32268099,
        1.1725733 , 3.81830505, 6.08152467, 2.71921475, 0.43057573,
        0.80370736, 3.07513022, 5.37650194, 2.21543445, 0.45562095],
 'VH': [0.65278359, 1.78216752, 2.51072492, 1.24251227, 0.2750489 ,
        0.5399724 , 1.87726985, 2.77666773, 1.24181038, 0.2784906 ,
        1.06999244, 3.29442887, 5.53889033, 2.55144416, 0.45256502,
        0.83927039, 3.15071258, 6.1677516 , 2.18943123, 0.3658004 ]}

cats_unique_NN = ['NN Cat A', 'NN Cat B', 'NN Cat C', 'NN Cat D']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters (edit as needed) ---
mass_bins = 5
processes_to_exclude = None  # or e.g. ["some_process"] or "some_process"

# --- First, define our processes and contributions for STXS ---
processes_STXS = list(combined_hist_STXS.keys())

# Exclude processes if needed
if processes_to_exclude:
    if isinstance(processes_to_exclude, str):
        processes_to_exclude = [processes_to_exclude]
    processes_STXS = [p for p in processes_STXS if p not in processes_to_exclude]

# Prepare STXS data
num_bins_STXS = len(cats_unique_STXS) * mass_bins
bin_indices_STXS = np.arange(num_bins_STXS)
contributions_STXS = np.array([combined_hist_STXS[proc] for proc in processes_STXS])

# --- Next, define our processes and contributions for NN ---
processes_NN = list(combined_hist_NN.keys())

# Exclude processes if needed
if processes_to_exclude:
    processes_NN = [p for p in processes_NN if p not in processes_to_exclude]

# Prepare NN data
num_bins_NN = len(cats_unique_NN) * mass_bins
bin_indices_NN = np.arange(num_bins_NN)
contributions_NN = np.array([combined_hist_NN[proc] for proc in processes_NN])

# --- For consistent colours across subplots, map each process to a unique colour ---
all_processes = list(set(processes_STXS + processes_NN))  # union of processes
palette = sns.color_palette("husl", len(all_processes))
process_to_colour = {proc: palette[i] for i, proc in enumerate(all_processes)}

# --- Create the figure with 2 vertical subplots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20))

# ===================
#  TOP SUBPLOT (STXS)
# ===================
bottom_STXS = np.zeros(num_bins_STXS)

for proc in processes_STXS:
    idx = processes_STXS.index(proc)  # index for contributions_STXS
    ax1.bar(
        bin_indices_STXS,
        contributions_STXS[idx],
        bottom=bottom_STXS,
        color=process_to_colour[proc],
        label=proc  # We'll collect legend handles from ax1
    )
    bottom_STXS += contributions_STXS[idx]

# X‐axis labels for STXS
x_labels_STXS = [
    f"{cat}" if i % mass_bins == (mass_bins // 2) else "" 
    for i, cat in enumerate(np.repeat(cats_unique_STXS, mass_bins))
]
ax1.set_xticks(bin_indices_STXS)
ax1.set_xticklabels(x_labels_STXS, rotation=0, fontsize=12)

# Vertical lines to separate categories
for i in range(1, len(cats_unique_STXS)):
    ax1.axvline(i * mass_bins - 0.5, color='black', linestyle='--', linewidth=0.5)

ax1.set_ylabel("Event Counts", fontsize=14)
ax1.set_title("Combined Histogram (STXS): Process Contributions", fontsize=16)

# =====================
#  BOTTOM SUBPLOT (NN)
# =====================
bottom_NN = np.zeros(num_bins_NN)

for proc in processes_NN:
    idx = processes_NN.index(proc)
    ax2.bar(
        bin_indices_NN,
        contributions_NN[idx],
        bottom=bottom_NN,
        color=process_to_colour[proc],
        label=proc  # We won't re‐use these labels; single legend from ax1 is enough
    )
    bottom_NN += contributions_NN[idx]

# X‐axis labels for NN
x_labels_NN = [
    f"{cat}" if i % mass_bins == (mass_bins // 2) else "" 
    for i, cat in enumerate(np.repeat(cats_unique_NN, mass_bins))
]
ax2.set_xticks(bin_indices_NN)
ax2.set_xticklabels(x_labels_NN, rotation=0, fontsize=12)

# Vertical lines to separate categories
for i in range(1, len(cats_unique_NN)):
    ax2.axvline(i * mass_bins - 0.5, color='black', linestyle='--', linewidth=0.5)

ax2.set_ylabel("Event Counts", fontsize=14)
ax2.set_xlabel("Categories", fontsize=14)
ax2.set_title("Combined Histogram (NN): Process Contributions", fontsize=16)

# =============
#   SINGLE LEGEND
# =============
handles, labels = ax1.get_legend_handles_labels()
# If you want each process to appear only once in the legend, we can do:
unique_labels = dict(zip(labels, handles))
fig.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=len(unique_labels),
    #title="Processes"
)

plt.tight_layout()
plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Assume these exist:
#   combined_hist_STXS, cats_unique_STXS
#   combined_hist_NN,   cats_unique_NN
#   mass_bins = 5       (or whatever number of bins)
# -----------------------------------------------------

# 1) Identify processes for STXS
processes_STXS = list(combined_hist_STXS.keys())
processes_STXS_no_bg = [p for p in processes_STXS if p.lower() != "background"]

# 2) Identify processes for NN
processes_NN = list(combined_hist_NN.keys())
processes_NN_no_bg = [p for p in processes_NN if p.lower() != "background"]

# 3) For consistent colours across all subplots, gather the union of all processes
all_processes = set(processes_STXS + processes_STXS_no_bg + processes_NN + processes_NN_no_bg)
all_processes = list(all_processes)  # for stable ordering

# Create a colour palette and map each process to a colour
palette = sns.color_palette("husl", len(all_processes))
process_to_colour = {proc: col for proc, col in zip(all_processes, palette)}

# 4) Prepare bin indices and contributions for each of the four variants
#    (STXS with BG, STXS no BG, NN with BG, NN no BG).
num_bins_STXS = len(cats_unique_STXS) * mass_bins
bin_indices_STXS = np.arange(num_bins_STXS)
contributions_STXS = {
    proc: np.array(combined_hist_STXS[proc]) for proc in processes_STXS
}
contributions_STXS_no_bg = {
    proc: np.array(combined_hist_STXS[proc]) for proc in processes_STXS_no_bg
}

num_bins_NN = len(cats_unique_NN) * mass_bins
bin_indices_NN = np.arange(num_bins_NN)
contributions_NN = {
    proc: np.array(combined_hist_NN[proc]) for proc in processes_NN
}
contributions_NN_no_bg = {
    proc: np.array(combined_hist_NN[proc]) for proc in processes_NN_no_bg
}

# 5) Create the 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
(ax1, ax2), (ax3, ax4) = axes

# --------------------------------------------------------------------------------
# Helper snippet to plot a single stacked‐bar subplot
def plot_stacked_hist(ax, bin_indices, processes, contrib_dict, categories, title):
    """
    ax: axes object
    bin_indices: np.arange(num_bins)
    processes: list of processes to be included
    contrib_dict: dict of {process_name: array_of_values}
    categories: e.g. cats_unique_STXS or cats_unique_NN
    title: subplot title
    """
    bottom = np.zeros_like(bin_indices, dtype=float)
    for proc in processes:
        ax.bar(
            bin_indices,
            contrib_dict[proc],
            bottom=bottom,
            color=process_to_colour[proc],
            edgecolor="none",
            label=proc
        )
        bottom += contrib_dict[proc]

    # Label the bins.  Show the category label in the *middle* of each group:
    x_labels = [
        f"{cat}" if i % mass_bins == mass_bins//2 else ""
        for i, cat in enumerate(np.repeat(categories, mass_bins))
    ]
    ax.set_xticks(bin_indices)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=18)

    # Vertical dashed lines to separate category regions
    for i in range(1, len(categories)):
        ax.axvline(i * mass_bins - 0.5, color='black', linestyle='--', linewidth=0.5)

    ax.set_title(title, fontsize=17)
    ax.set_ylabel("Event Counts", fontsize=14)

# --------------------------------------------------------------------------------
# 6) Top Left: STXS WITH background
plot_stacked_hist(
    ax1,
    bin_indices_STXS,
    processes_STXS,          # includes background
    contributions_STXS,
    cats_unique_STXS,
    "STXS (with background)"
)

# 7) Top Right: STXS WITHOUT background
plot_stacked_hist(
    ax2,
    bin_indices_STXS,
    processes_STXS_no_bg,    # excludes background
    contributions_STXS_no_bg,
    cats_unique_STXS,
    "STXS (no background)"
)

# 8) Bottom Left: NN WITH background
plot_stacked_hist(
    ax3,
    bin_indices_NN,
    processes_NN,            # includes background
    contributions_NN,
    cats_unique_NN,
    "NN (with background)"
)

# 9) Bottom Right: NN WITHOUT background
plot_stacked_hist(
    ax4,
    bin_indices_NN,
    processes_NN_no_bg,      # excludes background
    contributions_NN_no_bg,
    cats_unique_NN,
    "NN (no background)"
)
ax4.set_xlabel("Categories", fontsize=20)
ax3.set_xlabel("Categories", fontsize=20)

# --------------------------------------------------------------------------------
# 10) Single legend at the top (build from all processes in the top‐left subplot)
handles, labels = ax1.get_legend_handles_labels()

# If you want to ensure uniqueness (one handle per process name):
unique_labels = dict(zip(labels, handles))
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(unique_labels),
    #title="Processes"
)

#plt.tight_layout()

plt.savefig(thesis_plot_path + "/super_combined_hist.pdf")

plt.show()

#%%


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# -----------------------------------------------------
# Assume these exist:
#   combined_hist_STXS, cats_unique_STXS
#   combined_hist_NN,   cats_unique_NN
#   mass_bins = 5       (or whatever number of bins)
# -----------------------------------------------------

# 1) Identify processes for STXS
processes_STXS = list(combined_hist_STXS.keys())
processes_STXS_no_bg = [p for p in processes_STXS if p.lower() != "background"]

# 2) Identify processes for NN
processes_NN = list(combined_hist_NN.keys())
processes_NN_no_bg = [p for p in processes_NN if p.lower() != "background"]

# 3) For consistent colours across all subplots, gather the union of all processes
all_processes = set(processes_STXS + processes_STXS_no_bg + processes_NN + processes_NN_no_bg)
all_processes = list(all_processes)  # for stable ordering

# Create a colour palette and map each process to a colour
palette = sns.color_palette("husl", len(all_processes))
process_to_colour = {proc: col for proc, col in zip(all_processes, palette)}

# 4) Prepare bin indices and contributions for each of the four variants
#    (STXS with BG, STXS no BG, NN with BG, NN no BG).
num_bins_STXS = len(cats_unique_STXS) * mass_bins
bin_indices_STXS = np.arange(num_bins_STXS)
contributions_STXS = {
    proc: np.array(combined_hist_STXS[proc]) for proc in processes_STXS
}
contributions_STXS_no_bg = {
    proc: np.array(combined_hist_STXS[proc]) for proc in processes_STXS_no_bg
}

num_bins_NN = len(cats_unique_NN) * mass_bins
bin_indices_NN = np.arange(num_bins_NN)
contributions_NN = {
    proc: np.array(combined_hist_NN[proc]) for proc in processes_NN
}
contributions_NN_no_bg = {
    proc: np.array(combined_hist_NN[proc]) for proc in processes_NN_no_bg
}

# 5) Create the 2×2 figure with wider spacing between subplots to accommodate arrows
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
(ax1, ax2), (ax3, ax4) = axes

# --------------------------------------------------------------------------------
# Helper snippet to plot a single stacked‐bar subplot
def plot_stacked_hist(ax, bin_indices, processes, contrib_dict, categories, title):
    """
    ax: axes object
    bin_indices: np.arange(num_bins)
    processes: list of processes to be included
    contrib_dict: dict of {process_name: array_of_values}
    categories: e.g. cats_unique_STXS or cats_unique_NN
    title: subplot title
    """
    bottom = np.zeros_like(bin_indices, dtype=float)
    for proc in processes:
        ax.bar(
            bin_indices,
            contrib_dict[proc],
            bottom=bottom,
            color=process_to_colour[proc],
            edgecolor="none",
            label=proc
        )
        bottom += contrib_dict[proc]

    # Label the bins.  Show the category label in the *middle* of each group:
    x_labels = [
        f"{cat}" if i % mass_bins == mass_bins//2 else ""
        for i, cat in enumerate(np.repeat(categories, mass_bins))
    ]
    ax.set_xticks(bin_indices)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=12)

    # Vertical dashed lines to separate category regions
    for i in range(1, len(categories)):
        ax.axvline(i * mass_bins - 0.5, color='black', linestyle='--', linewidth=0.5)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Event Counts", fontsize=12)

# --------------------------------------------------------------------------------
# 6) Top Left: STXS WITH background
plot_stacked_hist(
    ax1,
    bin_indices_STXS,
    processes_STXS,          # includes background
    contributions_STXS,
    cats_unique_STXS,
    "STXS (with background)"
)

# 7) Top Right: STXS WITHOUT background
plot_stacked_hist(
    ax2,
    bin_indices_STXS,
    processes_STXS_no_bg,    # excludes background
    contributions_STXS_no_bg,
    cats_unique_STXS,
    "STXS (no background)"
)

# 8) Bottom Left: NN WITH background
plot_stacked_hist(
    ax3,
    bin_indices_NN,
    processes_NN,            # includes background
    contributions_NN,
    cats_unique_NN,
    "NN (with background)"
)

# 9) Bottom Right: NN WITHOUT background
plot_stacked_hist(
    ax4,
    bin_indices_NN,
    processes_NN_no_bg,      # excludes background
    contributions_NN_no_bg,
    cats_unique_NN,
    "NN (no background)"
)
ax4.set_xlabel("Categories", fontsize=14)
ax3.set_xlabel("Categories", fontsize=14)

# --------------------------------------------------------------------------------
# 10) Single legend at the top (build from all processes in the top‐left subplot)
handles, labels = ax1.get_legend_handles_labels()

# If you want to ensure uniqueness (one handle per process name):
unique_labels = dict(zip(labels, handles))

fig.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(unique_labels),
    #title="Processes"
)

# --------------------------------------------------------------------------------
# 11) Add large arrows with text between the plots

# Apply tight layout first to get the proper positioning of the plots
plt.tight_layout()

# Function to add an arrow with text between two axes
def add_arrow_with_text(fig, ax_left, ax_right, text, y_position):
    # Get the positions of the axes in figure coordinates
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()
    
    # Calculate arrow parameters
    arrow_width = (bbox_right.x0 - bbox_left.x1) * 0.9  # 90% of the space between plots
    arrow_x_start = bbox_left.x1 + (bbox_right.x0 - bbox_left.x1) * 0.05  # Start 5% after left plot
    arrow_y = y_position
    
    # Create black rectangle with an arrow at the right end
    rect_height = min(bbox_left.height, bbox_right.height) * 0.1  # 20% of plot height
    rect_width = arrow_width * 0.8  # Rectangle is 95% of arrow width
    
    # Create the black rectangle
    rect = patches.Rectangle(
        (arrow_x_start, arrow_y - rect_height/2),
        rect_width,
        rect_height,
        facecolor='black',
        alpha=0.8,
        transform=fig.transFigure,
        clip_on=False
    )
    
    # Create the arrowhead at the right end
    arrow_head = patches.Polygon(
        [
            (arrow_x_start + rect_width, arrow_y - rect_height/2),  # Bottom left
            (arrow_x_start + rect_width, arrow_y + rect_height/2),  # Top left
            (arrow_x_start + rect_width + rect_height, arrow_y),    # Right tip
        ],
        facecolor='black',
        alpha=0.8,
        transform=fig.transFigure,
        clip_on=False
    )
    
    # Add the shapes to the figure
    fig.patches.append(rect)
    fig.patches.append(arrow_head)
    
    # Split the text into two lines
    lines = text.split()
    line1 = lines[0]
    line2 = lines[1]
    
    # Add text in the middle of the arrow, stacked vertically
    text_x = arrow_x_start + rect_width / 2
    
    # Add both lines with white text on black background
    fig.text(text_x, arrow_y + rect_height*0.1, line1, 
             ha='center', va='center', 
             fontsize=11, fontweight='bold',
             color='white', transform=fig.transFigure)
    
    fig.text(text_x, arrow_y - rect_height*0.1, line2, 
             ha='center', va='center', 
             fontsize=11, fontweight='bold',
             color='white', transform=fig.transFigure)

# Add arrow between top plots (STXS)
y_top = (ax1.get_position().y0 + ax1.get_position().y1) / 2
add_arrow_with_text(fig, ax1, ax2, "Excluding Background", y_top)

# Add arrow between bottom plots (NN)
y_bottom = (ax3.get_position().y0 + ax3.get_position().y1) / 2
add_arrow_with_text(fig, ax3, ax4, "Excluding Background", y_bottom)

# Adjust the layout again to make room for the arrows
plt.subplots_adjust(wspace=0.3)  # Add more width space between subplots

plt.show()
