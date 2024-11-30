import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import *

procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF (x10)", "green"],
    "VH" : ["VH (x10)", "brown"]
}
col_name = "_sel"

def get_dfs(sample_path):
    dfs = {}
    for i, proc in enumerate(procs.keys()):
        #if proc != "ttH": continue
        print(f" --> Loading process: {proc}")
        dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

        # Remove nans from dataframe
        dfs[proc] = dfs[proc][(dfs[proc]['mass'+col_name] == dfs[proc]['mass'+col_name])]

        # Reweight to target lumi 
        dfs[proc]['plot_weight'] = dfs[proc]['plot_weight']*(target_lumi/total_lumi)

        # Calculate true weight: remove x10 multiplier for signal
        if proc in ['ggH', 'VBF', 'VH', 'ttH']:
            dfs[proc]['true_weight'+col_name] = dfs[proc]['plot_weight']/10
        else:
            dfs[proc]['true_weight'+col_name] = dfs[proc]['plot_weight']

        # Add variables
        b_tag_scores = np.array(dfs[proc][['j0_btagB'+col_name, 'j1_btagB'+col_name, 'j2_btagB'+col_name, 'j3_btagB'+col_name]])
        b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
        max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
        second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
        # Add nans back in for plotting tools below
        max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
        second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
        dfs[proc]['max_b_tag_score'] = max_b_tag_score
        dfs[proc]['second_max_b_tag_score'] = second_max_b_tag_score
    return dfs

def get_selection(dfs):
    for i, proc in enumerate(procs.keys()):   
        yield_before_sel = dfs[proc]['true_weight'+col_name].sum()
        mask = dfs[proc]['n_jets'+col_name] >= 2
        mask = mask & (dfs[proc]['max_b_tag_score'] > 0.7)
        mask = mask & (dfs[proc]['second_max_b_tag_score'] > 0.4)
        dfs[proc] = dfs[proc][mask]
        yield_after_sel = dfs[proc]['true_weight'+col_name].sum()
        eff = (yield_after_sel/yield_before_sel)*100
        print(f"{proc}: N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")
    return dfs

        