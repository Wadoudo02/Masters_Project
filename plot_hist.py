import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from analysis.tth_eft.Avighna.utils import *

# Constants
total_lumi = 7.9804

# Normalise plots to unity: compare shapes
plot_fraction = bool(int(sys.argv[2]))

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF (x10)", "green"],
    "VH" : ["VH (x10)", "brown"]
}

# Vars to plot
#vars_to_plot = sys.argv[1].split(",")
vars_to_plot = ['plot_weight', 'mass', 'minIDMVA-with-cut', 'n_jets', 'lead_pixelSeed', 'sublead_pixelSeed', 'deltaR', 'delta_eta_gg', 'delta_phi_gg', 'delta_eta_jj', 'delta_phi_jj', 'dijet_mass', 'delta_phi_gg_jj', 'min_delta_R_j_g', 'lead_pt/mass', 'sublead_pt/mass', 'pt/mass', 'rapidity', 'lead_eta', 'sublead_eta', 'lead_mvaID', 'sublead_mvaID', 'lead_phi', 'sublead_phi', 'lead_pt', 'sublead_pt', 'lead_r9', 'sublead_r9', 'sigma_m_over_m', 'lead_hoe', 'sublead_hoe', 'PVScore', 'MET_pt', 'MET_phi', 'HT', 'j0_pt', 'j1_pt', 'j2_pt', 'j3_pt', 'j0_eta', 'j1_eta', 'j2_eta', 'j3_eta', 'j0_phi', 'j1_phi', 'j2_phi', 'j3_phi', 'j0_btagB', 'j1_btagB', 'j2_btagB', 'j3_btagB', 'Ele0_pt', 'Ele1_pt', 'Ele0_eta', 'Ele1_eta', 'Ele0_phi', 'Ele1_phi', 'Ele0_charge', 'Ele1_charge', 'Ele0_id', 'Ele1_id', 'n_electrons', 'Muo0_pt', 'Muo1_pt', 'Muo0_eta', 'Muo1_eta', 'Muo0_phi', 'Muo1_phi', 'Muo0_charge', 'Muo1_charge', 'Muo0_id', 'Muo1_id', 'n_muons', 'Tau0_pt', 'Tau1_pt', 'Tau0_eta', 'Tau1_eta', 'Tau0_phi', 'Tau1_phi', 'Tau0_charge', 'Tau1_charge', 'n_taus', 'Lep0_pt', 'Lep1_pt', 'Lep0_eta', 'Lep1_eta', 'Lep0_phi', 'Lep1_phi', 'Lep0_charge', 'Lep1_charge', 'Lep0_flav', 'Lep1_flav', 'n_leptons', 'cosDeltaPhi', 'deltaPhiJ0GG', 'deltaPhiJ1GG', 'deltaPhiJ2GG', 'deltaEtaJ0GG', 'deltaEtaJ1GG', 'deltaEtaJ2GG', 'centrality', 'dilepton_mass', 'deltaR_L0G0', 'deltaR_L0G1', 'deltaR_L1G0', 'deltaR_L1G1', 'theta_ll_gg', 'cosThetaStar', 'deltaPhiMetGG', 'minDeltaPhiJMET', 'pt_balance', 'helicity_angle']
# Load dataframes
dfs = {}
for i, proc in enumerate(procs.keys()):
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")

    if i==0:
        print(" --> Columns in first dataframe: ", list(dfs[proc].columns))

    # Remove nans from dataframe
    dfs[proc] = dfs[proc][(dfs[proc]['mass'] == dfs[proc]['mass'])]

    # Add additional variables
    # Example: (second-)max-b-tag score
    b_tag_scores = np.array(dfs[proc][['j0_btagB', 'j1_btagB', 'j2_btagB', 'j3_btagB']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    # Add nans back in for plotting tools below
    max_b_tag_score = np.where(max_b_tag_score==-1, np.nan, max_b_tag_score)
    second_max_b_tag_score = np.where(second_max_b_tag_score==-1, np.nan, second_max_b_tag_score)
    dfs[proc]['max_b_tag_score'] = max_b_tag_score
    dfs[proc]['second_max_b_tag_score'] = second_max_b_tag_score


fig, ax = plt.subplots(1,1)
for v in vars_to_plot:
    print(f" --> Plotting: {v}")
    try:
        nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
    except KeyError:
        nbins, xrange, is_log_scale, sanitized_var_name = 50, (0, 50), False, v
    # Loop over procs and add histogram
    for proc in procs.keys():
        label, color = procs[proc]
        if plot_fraction:
            label = label.split(" ")[0]

        x = np.array(dfs[proc][v])
        
        # Remove nans from array and calculate fraction of events which have real value
        mask = x==x
        pc = 100*mask.sum()/len(x)
        if pc != 100:
            # Add information to label
            label += f", {pc:.1f}% plotted"
        x = x[mask]

        # Event weight
        w = np.array(dfs[proc]['plot_weight'])[mask]
        if plot_fraction:
            w /= w.sum()

        #ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)
        ax.hist(x, nbins, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

    ax.set_xlabel(sanitized_var_name)
    if plot_fraction:
        ax.set_ylabel("Fraction of events")
    else:
        ax.set_ylabel("Events")
    
    if is_log_scale:
        ax.set_yscale("log")

    ax.legend(loc='best')
    
    hep.cms.label("", year="2022 (preEE)", com="13.6", lumi=total_lumi, lumi_format="{0:.2f}", ax=ax)

    plt.tight_layout()
    ext = "_norm" if plot_fraction else ""
    #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
    fig.savefig(f"plots_all_prod/{v}{ext}.png", bbox_inches="tight")
    ax.cla()
