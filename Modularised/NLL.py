"""
NLL

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

from utils import *

# Function to extract 2NLL from array of NLL values
def TwoDeltaNLL(x):
    x = np.array(x)
    return 2*(x-x.min())



def calc_NLL(combined_histogram, mus, signal='ttH'):
    """
    Calculate the NLL using the combined 25-bin histogram with variable \mu parameters.

    Parameters:
        combined_histogram (dict): Combined histogram for each process across 25 bins.
        mus (list or array): Signal strength modifiers, one for each truth category.
        signal (str): The signal process (default 'ttH').

    Returns:
        float: Total NLL.
    """
    NLL_total = 0.0
    num_bins = len(next(iter(combined_histogram.values())))  # Total bins (should be 25)

    # Loop over each bin in the combined histogram
    for bin_idx in range(num_bins):
        expected_total = 0.0
        observed_count = 0.0
        #breakpoint()
        for proc, yields in combined_histogram.items():
            if signal in proc:
                # Extract the truth category index from the signal label, e.g., "ttH_0"
                truth_cat_idx = int(proc.split('_')[1])
                mu = mus[truth_cat_idx]  # Apply the appropriate \mu for this truth category
                expected_total += mu * yields[bin_idx]
            else:
                expected_total += yields[bin_idx]

            observed_count += yields[bin_idx]  # Observed count from all processes in this bin

        # Avoid division by zero in log calculation
        expected_total = max(expected_total, 1e-10)
        
        # Calculate NLL contribution for this bin
        NLL_total += observed_count * np.log(expected_total) - expected_total

    return -NLL_total


def calc_Hessian_NLL(combined_histogram, mus, signal='ttH'):
    """
    Calculate the Hessian matrix for signal strength parameters.
    
    Parameters:
        combined_histogram (dict): Combined histogram for each process across bins.
        mus (list or array): Signal strength modifiers for each truth category.
        signal (str): The signal process (default 'ttH').
    
    Returns:
        numpy.ndarray: Hessian matrix for signal strength parameters.
    """
    # Determine the number of signal truth categories
    signal_categories = [proc for proc in combined_histogram.keys() if proc.startswith(signal + '_')]
    num_categories = len(signal_categories)
    
    # Initialize Hessian matrix
    Hessian = np.zeros((num_categories, num_categories))
    
    # Total number of bins
    num_bins = len(next(iter(combined_histogram.values())))  # Assume all histograms have the same length
    
    # Calculate Hessian matrix elements
    for i in range(num_categories):
        for m in range(num_categories):
            hessian_sum = 0.0
            
            for bin_idx in range(num_bins):
                # Get background contribution for this bin
                background = sum(
                    combined_histogram[proc][bin_idx]
                    for proc in combined_histogram
                    if not proc.startswith(signal + '_')
                )
                
                # Signal yields for this bin
                s_i = combined_histogram[signal_categories[i]][bin_idx]
                s_m = combined_histogram[signal_categories[m]][bin_idx]
                
                # Expected total: scaled signal + background
                expected_total = sum(
                    mus[k] * combined_histogram[signal_categories[k]][bin_idx]
                    for k in range(num_categories)
                ) + background
                
                # Ensure positive expected total to avoid division by zero
                expected_total = max(expected_total, 1e-10)
                
                # Total observed count in this bin
                observed_count = sum(combined_histogram[proc][bin_idx] for proc in combined_histogram)
                
                # Hessian contribution for this bin
                hessian_sum += (observed_count * s_i * s_m) / (expected_total ** 2)
            
            Hessian[i, m] = hessian_sum
    
    return Hessian