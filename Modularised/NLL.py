"""
NLL

@author: wadoudcharbak
"""
import numpy as np


import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from scipy.optimize import minimize


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

def objective_function(mus, fixed_mu_index, fixed_mu_value, combined_histogram):
    """
    Objective function to compute NLL for given mu values, fixing one mu.
    
    Parameters:
        mus (array-like): Array of 4 mu values to optimize.
        fixed_mu_index (int): Index of the mu being scanned (fixed during optimization).
        fixed_mu_value (float): The fixed value for the scanned mu.
    
    Returns:
        float: NLL value for the given set of mu values.
    """
    full_mus = np.insert(mus, fixed_mu_index, fixed_mu_value)  # Reconstruct full mu array
    return calc_NLL(combined_histogram, full_mus, signal='ttH')


def perform_NLL_scan_and_profile(
    mu_values,
    mus_initial,
    bounds,
    combined_histogram,
    signal='ttH',
    plot=False
):
    """
    Performs NLL scans and profiles for given mu parameters.
    
    Parameters:
        mu_values (array-like): Range of mu values for scanning.
        mus_initial (list): Initial guesses for the mu parameters.
        bounds (list): Bounds for the mu parameters during optimization.
        combined_histogram (Dictionary): Histogram data used for NLL calculations.
        signal (str): Signal type for the NLL calculation.
        plot (bool): Whether to plot the results.

    Returns:
        The optimised Mus from the Frozen and Profile scan
    """
    #breakpoint()
    frozen_mus = mus_initial.copy()
    frozen_optimised_mus = []
    profile_optimised_mus = []

    if plot:
        fig, axes = plt.subplots(nrows=5, figsize=(8, 30), dpi=300, sharex=True)

    for i in range(len(mus_initial)):
        frozen_NLL_vals = []
        profile_NLL_vals = []

        for mu in mu_values:
            # Frozen scan: keep other mu values constant
            frozen_mus[i] = mu
            frozen_NLL_vals.append(calc_NLL(combined_histogram, frozen_mus, signal=signal))
            
            # Profile scan: optimize the other mu values
            initial_guess = [mus_initial[j] for j in range(5) if j != i]
            obj_func = lambda reduced_mus: objective_function(reduced_mus, fixed_mu_index=i, fixed_mu_value=mu, combined_histogram = combined_histogram)
            result = minimize(obj_func, initial_guess, bounds=bounds, method='L-BFGS-B')
            profile_NLL_vals.append(result.fun)
            
        
        # Convert to 2ΔNLL
        frozen_NLL_vals = TwoDeltaNLL(frozen_NLL_vals)
        profile_NLL_vals = TwoDeltaNLL(profile_NLL_vals)

        # Find crossings
        frozen_vals = find_crossings((mu_values, frozen_NLL_vals), 1.0)
        profile_vals = find_crossings((mu_values, profile_NLL_vals), 1.0)
        frozen_label = add_val_label(frozen_vals)
        profile_label = add_val_label(profile_vals)

        # Update optimal frozen mu
        frozen_mus[i] = frozen_vals[0][0]

        # Store results
        frozen_optimised_mus.append(frozen_vals[0][0])
        profile_optimised_mus.append(profile_vals[0][0])

        if plot:
            # Plotting each NLL curve on a separate subplot
            ax = axes[i]
            ax.plot(mu_values, frozen_NLL_vals, label=f"Frozen Scan: {frozen_label}", color='blue')
            ax.plot(mu_values, profile_NLL_vals, label=f"Profile Scan: {profile_label}", color='red', linestyle='--')
            ax.axvline(1.0, label="SM (expected)", color='green', alpha=0.5)
            ax.axhline(1.0, color='grey', alpha=0.5, ls='--')
            ax.axhline(4.0, color='grey', alpha=0.5, ls='--')
            ax.set_ylim(0, 8)
            ax.legend(loc='best')
            ax.set_ylabel("q = 2$\\Delta$NLL")
            ax.set_title(f"Optimizing $\\mu_{i}$")

    if plot:
        # Show the plot
        plt.xlabel("$\\mu$ Value")
        plt.tight_layout()
        plt.show()

    return frozen_optimised_mus, profile_optimised_mus


def perform_NLL_scan_and_profile_and_other_mus(
    mu_values,
    mus_initial,
    bounds,
    combined_histogram,
    signal='ttH',
    plot=False
):
    """
    Performs NLL scans and profiles for given mu parameters.
    
    Parameters:
        mu_values (array-like): Range of mu values for scanning.
        mus_initial (list): Initial guesses for the mu parameters.
        bounds (list): Bounds for the mu parameters during optimization.
        combined_histogram (Dictionary): Histogram data used for NLL calculations.
        signal (str): Signal type for the NLL calculation.
        plot (bool): Whether to plot the results.
    Returns:
        The optimised Mus from the Profile scan
    """
    frozen_mus = mus_initial.copy()
    frozen_optimised_mus = []
    profile_optimised_mus = []
    
    # Store minimized other mus for plotting
    other_minimised_mus_profile = [[] for _ in range(len(mus_initial))]
    
    if plot:
        # Create a 5x2 subplot grid
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 30), dpi=300)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i in range(len(mus_initial)):
        frozen_NLL_vals = []
        profile_NLL_vals = []
        
        # Temporary storage for profile-scanned other mus
        current_other_minimised_mus_profile = [[] for _ in range(len(mus_initial)-1)]
        
        for mu in mu_values:
            # Frozen scan: keep other mu values constant
            frozen_mus[i] = mu
            frozen_NLL_vals.append(calc_NLL(combined_histogram, frozen_mus, signal=signal))
            
            # Profile scan: optimize the other mu values
            initial_guess = [mus_initial[j] for j in range(len(mus_initial)) if j != i]
            obj_func = lambda reduced_mus: objective_function(reduced_mus, fixed_mu_index=i, fixed_mu_value=mu, combined_histogram=combined_histogram)
            result = minimize(obj_func, initial_guess, bounds=bounds, method='L-BFGS-B')
            profile_NLL_vals.append(result.fun)
            
            # Store the profile-scanned other mus
            profile_other_mus = list(result.x)
            for k, other_mu in enumerate(profile_other_mus):
                current_other_minimised_mus_profile[k].append(other_mu)
        
        # Convert to 2ΔNLL
        frozen_NLL_vals = TwoDeltaNLL(frozen_NLL_vals)
        profile_NLL_vals = TwoDeltaNLL(profile_NLL_vals)
        
        # Find crossings
        frozen_vals = find_crossings((mu_values, frozen_NLL_vals), 1.0)
        profile_vals = find_crossings((mu_values, profile_NLL_vals), 1.0)
        
        frozen_label = add_val_label(frozen_vals)
        profile_label = add_val_label(profile_vals)
        
        # Update optimal frozen mu
        frozen_mus[i] = frozen_vals[0][0]
        
        # Store results
        frozen_optimised_mus.append(frozen_vals[0][0])
        profile_optimised_mus.append(profile_vals[0][0])
        
        if plot:
            # NLL plot on the left column
            ax_nll = axes[i, 0]
            ax_nll.plot(mu_values, frozen_NLL_vals, label=f"Frozen Scan: {frozen_label}", color='blue')
            ax_nll.plot(mu_values, profile_NLL_vals, label=f"Profile Scan: {profile_label}", color='red', linestyle='--')
            ax_nll.axvline(1.0, label="SM (expected)", color='green', alpha=0.5)
            ax_nll.axhline(1.0, color='grey', alpha=0.5, ls='--')
            ax_nll.axhline(4.0, color='grey', alpha=0.5, ls='--')
            ax_nll.set_ylim(0, 8)
            ax_nll.legend(loc='best')
            ax_nll.set_ylabel("q = 2$\\Delta$NLL")
            ax_nll.set_title(f"Optimizing $\\mu_{{{i}}}$ - NLL")
            
            # Other mus plot on the right column
            ax_other_mus = axes[i, 1]
            
            # Plot profile scan values of other mus
            for k, other_mu_vals in enumerate(current_other_minimised_mus_profile):
                ax_other_mus.plot(mu_values, other_mu_vals, 
                                  label=f"Profile $\\mu_{{{k if k < i else k+1}}}$", 
                                  color=plt.cm.Set1(k), linestyle='--')
            
            ax_other_mus.axvline(1.0, label="SM (expected)", color='green', alpha=0.5)
            ax_other_mus.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
            ax_other_mus.set_ylabel("Other $\\mu$ Values")
            ax_other_mus.set_title(f"Optimizing $\\mu_{{{i}}}$ - Other $\\mu$ Values")
    
    if plot:
        # Final x-label for the bottom row
        axes[-1, 0].set_xlabel("$\\mu$ Value")
        axes[-1, 1].set_xlabel("$\\mu$ Value")
        plt.tight_layout()
        plt.show()
    
    return frozen_optimised_mus, profile_optimised_mus