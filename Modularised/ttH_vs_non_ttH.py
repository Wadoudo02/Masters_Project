import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from utils import *

plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300
plot_fraction = True  # Toggle for normalizing histograms to fractions

# Processes to plot
procs = {
    #"background": ["Background", "black"],
    "ggH": ["ggH", "cornflowerblue"],
    "VBF": ["VBF", "red"],
    "VH": ["VH", "orange"],
    "ttH": ["ttH", "mediumorchid"],
}



# Variable to plot
v = "j3_btagB"

for v in vars_plotting_dict.keys():
    
    if v == "plot_weight":
        continue

    # Extract plotting details from vars_plotting_dict
    if v == "pt":
        num_bins, plot_range, logplot, x_label = [50, (0, 1000), False, "$p_T$"]
    else:
        num_bins, plot_range, logplot, x_label = vars_plotting_dict[v]
    
    v += "_sel"
    
    # Load and preprocess data for all processes
    dfs = {}
    for proc, (label, color) in procs.items():
        print(f" --> Loading process: {proc}")
        dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")
        dfs[proc] = dfs[proc][(dfs[proc]["mass_sel"] == dfs[proc]["mass_sel"])]  # Remove NaNs
        dfs[proc]['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
        dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Plot Standard Model (SM) histograms
    ttH_data = dfs["ttH"].copy()
    non_ttH_data = pd.concat(
        [dfs[proc] for proc in procs if proc != "ttH"], ignore_index=True
    )
    
    # Histogram data for ttH
    x_ttH = np.array(ttH_data[v])
    w_ttH = np.array(ttH_data["plot_weight"])
    
    ax.hist(
        x_ttH,
        bins=num_bins,
        range=plot_range,
        density=plot_fraction,  
        weights=w_ttH,
        histtype="step",
        color="mediumorchid",
        linewidth=2,
        alpha=1,
        label="ttH",
    )
    
    # Histogram data for non-ttH
    x_non_ttH = np.array(non_ttH_data[v])
    w_non_ttH = np.array(non_ttH_data["plot_weight"])
    
    ax.hist(
        x_non_ttH,
        bins=num_bins,
        range=plot_range,
        density=plot_fraction,  
        weights=w_non_ttH,
        histtype="step",
        color="black",
        linewidth=2,
        alpha=1,
        label="non-ttH",
    )
    
    # Label and formatting
    ax.set_ylabel("Events" if not plot_fraction else "Fraction of Events")
    if logplot:
        ax.set_yscale("log")
    hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
    ax.legend(loc="best", ncol=1)
    
    # Shared x-axis label from vars_plotting_dict
    ax.set_xlabel(x_label)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    # fig.savefig(f"{plot_path}/ttH_vs_non_ttH_{v}.png", bbox_inches="tight")
    plt.show()