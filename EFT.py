#%%
from utils import *
from selection import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ttH_df = pd.read_parquet(f"{new_sample_path}/ttH_processed_selected_with_smeft_cut_mupcleq90.parquet")
ttH_df = prep_df(ttH_df, "ttH")

ttH_df = get_selection(ttH_df, "ttH")

print(list(ttH_df.columns))
#%%
'''
Best fit for c_g: -0.01
68% confidence interval for c_g: (np.float64(-0.3103103103103102), np.float64(0.2702702702702702))
Best fit for c_tg: 0.01
68% confidence interval for c_tg: (np.float64(-0.41041041041041026), np.float64(0.6506506506506504))
'''
c_g_con = 0.27
c_tg_con = 0.65
n_bins = 40
#ax.hist(ttH_df["mass_sel"], bins=100, label="cg=0, ctg=0", weights=ttH_df["plot_weight"])

def get_features():
    return list(ttH_df.columns)

#Can provide a single value of cg and ctg for all events or a list of values for each event
def calc_weights(df, cg=c_g_con, ctg=c_tg_con, weight_col = "plot_weight"):
    cur_weights=df[weight_col]*(1+df["a_cg"]*cg +
                                    df["a_ctgre"]*ctg +
                                    df["b_cg_cg"]*(cg**2 if not isinstance(cg, (list, np.ndarray)) else [i**2 for i in cg]) +
                                    df["b_cg_ctgre"]*ctg*cg + 
                                    df["b_ctgre_ctgre"]*(ctg**2 if not isinstance(ctg, (list, np.ndarray)) else [i**2 for i in ctg]))
    return cur_weights
def apply_weight_change(df, ax, cg=0, ctg=0, var="mass_sel"):
    
    cur_weights=calc_weights(df, cg=cg, ctg=ctg)
    if cg==0 and ctg==0:
        hist = ax.hist(df[var], bins=n_bins, label=f"cg={cg}, ctg={ctg}, SM", weights=cur_weights, alpha = 0.3, color="gray", density=True)
    else:
        hist = ax.hist(df[var], bins=n_bins, label=f"cg={cg}, ctg={ctg}", weights=cur_weights, histtype="step",linewidth=1.5,density=True)
    return hist
    
#All_combs =True will plot all 4 combinations, False will plot SM and just cg and just ctg combs aswell
def plot_eft_hists(var="mass_sel", all_combs = True): 
    fig, (ax, ax_ratio) = plt.subplots(2,1, figsize=(15,15),gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    #ax.set_xlim(0,500)
    ax.legend()
    ax.set_xlabel(f"{var}")
    ax.set_ylabel("Events")

    fig.suptitle("Event distribution for different wilson coefficients.")

    if all_combs:
        combs = [(c_g_con, c_tg_con), (0, c_tg_con), (c_g_con, 0)]
    else:
        combs = [(c_g_con, c_tg_con)]
    hist_sm = apply_weight_change(ttH_df, ax, cg=0, ctg=0, var=var)

    for c_g, c_tg in combs:
        hist = apply_weight_change(ttH_df, ax, cg=c_g, ctg=c_tg, var=var)

        bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
        ratio = hist[0] / hist_sm[0]

        ax_ratio.plot(bin_centers, ratio, label=f"cg={c_g}, ctg={c_tg}", drawstyle='steps-mid')

    ax_ratio.set_ylim(0,7)

    ax_ratio.axhline(1, color='grey', linestyle='--')
    ax_ratio.set_ylabel("Ratio to SM")
    ax_ratio.legend()

    plt.tight_layout()
    plt.show()

def plot_SMEFT_features(features):
    for feat in features:
        print(feat)
        plot_eft_hists(var=feat, all_combs = False)


# %%
