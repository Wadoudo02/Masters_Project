#%%
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ttH_df = pd.read_parquet(f"{new_sample_path}/ttH_processed_selected_with_smeft.parquet")
ttH_df["plot_weight"] = ttH_df["plot_weight"]*(target_lumi/total_lumi)

print(list(ttH_df.columns))
#%%
'''
Best fit for c_g: -0.01
68% confidence interval for c_g: (np.float64(-0.4904904904904903), np.float64(0.3303303303303302))
Best fit for c_tg: 0.01
68% confidence interval for c_tg: (np.float64(-0.4304304304304303), np.float64(0.8908908908908906))
'''

#ax.hist(ttH_df["mass_sel"], bins=100, label="cg=0, ctg=0", weights=ttH_df["plot_weight"])

def apply_weight_change(df, ax, cg=0, ctg=0, var="mass_sel"):
    
    cur_weights=df["plot_weight"]*(1+df["a_cg"]*cg +
                                    df["a_ctgre"]*ctg +
                                    df["b_cg_cg"]*cg**2 +
                                    df["b_cg_ctgre"]*ctg*cg + 
                                    df["b_ctgre_ctgre"]*ctg**2)
    plt.hist(df[var], bins=100, label=f"cg={cg}, ctg={ctg}", weights=cur_weights, alpha=0.5)
    ax.set_xlim(0, 200)
    ax.legend()
    ax.set_xlabel(f"{var}")
    ax.set_ylabel("Events")

def plot_eft_hists(var="mass_sel"):
    fig, ax = plt.subplots(figsize=(15,7),sharex=True, sharey=True)

    apply_weight_change(ttH_df, ax, cg=0.33, ctg=0.89, var=var)
    apply_weight_change(ttH_df, ax, cg=0, ctg=0.89, var=var)
    apply_weight_change(ttH_df, ax, cg=0.33, ctg=0, var=var)
    apply_weight_change(ttH_df, ax, cg=0, ctg=0, var=var)

    fig.suptitle("Event distribution for different wilson coefficients.")

plot_eft_hists(var="HTXS_Higgs_pt_sel")


#%%