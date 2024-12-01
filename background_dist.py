#%%
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from utils import get_pt_cat

def exp(x, lam, A):
    return A*np.exp(-lam*(x-100))
def get_background_dist(back_data, num_cats=5):
    back_mass = back_data["mass_sel"]#.dropna().reset_index(drop=True)
    back_pt = back_data["pt-over-mass_sel"]*back_mass
    #print(back_pt)
    back_data["categories"] = get_pt_cat(back_pt)
    back_data["categories"].dropna()
    #print(back_data["categories"])
    fig,ax = plt.subplots(ncols=5, figsize=(20,5))
    #print(np.unique(back_data["categories"]))
    num_bins = 50
    fits = []
    for cat in range(num_cats):
        mask = back_data["categories"]==cat
        cur_cat = back_mass[mask]
        ax[cat].hist(cur_cat, bins=num_bins, range=(100,180), weights = back_data["plot_weight"][mask])
        counts, bin_edges = np.histogram(cur_cat, bins = num_bins, range= (100,180), weights = back_data["plot_weight"][mask])
        bin_centres = (bin_edges[:-1]+bin_edges[1:])/2

        #Ignoring 0s and outliers
        fil_cen = bin_centres[counts>0]
        fil_counts = counts[counts>0]

        
        p_fit, p_cov = curve_fit(exp, fil_cen, fil_counts,p0=[0.001, 10000])
        
        ax[cat].plot(np.arange(100, 180), exp(np.arange(100, 180), *p_fit))
        ax[cat].plot(fil_cen, fil_counts, linestyle="", marker="x")
        ax[cat].axvline(120, color="green")
        ax[cat].axvline(130, color="green")

        ax[cat].set_xlabel("Mass (GeV)")
        ax[cat].set_ylabel("Events")
        ax[cat].set_title(f"Category {cat}")
        fits.append(p_fit)
    #fig.savefig(f"{analysis_path}/back_mass.png", bbox_inches="tight")
    return fits

def get_back_int(data, cat, bounds, n_bins, num_cats=5):
    p_fit = get_background_dist(data,num_cats)[cat]
    lam, A = p_fit

    events = []
    bin_width = (bounds[1]-bounds[0])/n_bins

    for i in range(n_bins):
        lower_bound = bounds[0] + i * bin_width
        upper_bound = lower_bound + bin_width
        print("Integral bounds: ", lower_bound, upper_bound)
        integral = quad(exp, lower_bound, upper_bound, args=(lam, A))
        print("Integral: ", integral)
        events.append(integral[0])
    return np.array(events)



#get_background_dist()
#print(get_back_int(3, (120, 130), 5))

# %%
