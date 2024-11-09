
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def exp(x, lam, A):
    return A*np.exp(-lam*x)
def get_background_dist():
    back_data = pd.read_parquet(f"{sample_path}/Data_processed_selected.parquet")
    back_mass = back_data["mass_sel"][back_data["mass_sel"]==back_data["mass_sel"]]

    fig,ax = plt.subplots()
    
    counts, bin_edges = np.histogram(back_mass, bins = 80)
    bin_centres = (bin_edges[:-1]+bin_edges[1:])/2

    mask = (bin_centres<125) | (bin_centres>135)

    fil_cen = bin_centres[counts>0]
    fil_counts = counts[counts>0]
    
    p_fit, p_cov = curve_fit(exp, fil_cen, fil_counts,p0=[0.001, 10000])
    
    ax.plot(np.arange(100, 180), exp(np.arange(100, 180), *p_fit))
    ax.plot(fil_cen, fil_counts, linestyle="", marker="x")
    
    fig.savefig(f"{analysis_path}/back_mass.png", bbox_inches="tight")
    return p_fit




get_background_dist()
