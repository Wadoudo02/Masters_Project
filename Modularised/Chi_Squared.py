
"""
Chi Squared

@author: wadoudcharbak
"""

import json
import numpy as np


import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from scipy.optimize import minimize


from utils import *

ttH_coefficients_data = json.load(open(f"{sample_path}/TTH_standalone.json", 'r'))


optimized_mus = np.array([1,1,1,1,1])

a_cg = ttH_coefficients_data['data']["central"]["a_cg"]
a_ctgre = ttH_coefficients_data['data']["central"]["a_ctgre"]

b_cg_cg = ttH_coefficients_data['data']["central"]["b_cg_cg"]
b_cg_ctgre = ttH_coefficients_data['data']["central"]["b_cg_ctgre"]
b_ctgre_ctgre = ttH_coefficients_data['data']["central"]["b_ctgre_ctgre"]

# CHANGE TO C_G AND C_TG
def mu_c(cg, ctg, quadratic = True, num_of_categories = 5):
    """Theoretical model for mu as a function of Wilson coefficient c."""
    
    mus = []
    
    for i in range(1, (num_of_categories + 1)):
        mus.append(1 + cg * a_cg[i] + ctg * a_ctgre[i])

    if quadratic:
        for i in range(1, (num_of_categories + 1)):
            for j in range(num_of_categories):
                mus[j] += (cg ** 2) * b_cg_cg[i] + cg * ctg * b_cg_ctgre[i] + (ctg ** 2) * b_ctgre_ctgre[i]
        
    return np.array(mus)


def chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order):
    # Extract scalar values if arrays are passed
    cg_val = cg[0] if isinstance(cg, np.ndarray) else cg
    ctg_val = ctg[0] if isinstance(ctg, np.ndarray) else ctg
    
    delta_mu = optimized_mus - mu_c(cg_val, ctg_val, quadratic_order)
    return float(delta_mu.T @ hessian_matrix @ delta_mu)


def find_confidence_interval(chi_2, c_vals, min_chi_2, delta_chi_2): # Copied over from Avighna - if you are reading this thank you :) 
    lower_bound = None
    upper_bound = None
    for i, chi in enumerate(chi_2):
        if chi <= min_chi_2 + delta_chi_2:
            if lower_bound is None:
                lower_bound = c_vals[i]
            upper_bound = c_vals[i]
    return [min_chi_2, lower_bound, upper_bound]


    
    
def chi_squared_scans(
    optimized_mus,
    hessian_matrix,
    range_of_values,
    quadratic_order=True,
    fixed_ctg=0,
    fixed_cg=0,
):
    cg_values = range_of_values.copy()
    ctg_values = range_of_values.copy()

    profile_chi_squared_cg = []
    profile_chi_squared_ctg = []
    minimized_ctg_for_cg = []  # Store minimized c_tg values for each c_g
    minimized_cg_for_ctg = []  # Store minimized c_g values for each c_tg

    order = "Quadratic" if quadratic_order else "First"

    # Profile scans for c_g
    for cg in cg_values:
        result = minimize(lambda ctg: chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order), x0=0)
        profile_chi_squared_cg.append(result.fun)
        minimized_ctg_for_cg.append(result.x[0])

    # Profile scans for c_tg
    for ctg in ctg_values:
        result = minimize(lambda cg: chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order), x0=0)
        profile_chi_squared_ctg.append(result.fun)
        minimized_cg_for_ctg.append(result.x[0])

    # Frozen scans    

    frozen_chi_squared_cg = [chi_squared_func(cg, fixed_ctg, optimized_mus, hessian_matrix, quadratic_order) for cg in cg_values]
    frozen_chi_squared_ctg = [chi_squared_func(fixed_cg, ctg, optimized_mus, hessian_matrix, quadratic_order) for ctg in ctg_values]

    
    # Add Labels for cg
    
    frozen_cg_vals = find_confidence_interval(frozen_chi_squared_cg, cg_values, min(frozen_chi_squared_cg), 1)
    profile_cg_vals = find_confidence_interval(profile_chi_squared_cg, cg_values, min(profile_chi_squared_cg), 1)
    
    frozen_cg_label = add_val_label(frozen_cg_vals)
    profile_cg_label = add_val_label(profile_cg_vals)
    
    # Add Labels for ctg
    
    frozen_ctg_vals = find_confidence_interval(frozen_chi_squared_ctg, ctg_values, min(frozen_chi_squared_ctg), 1)
    profile_ctg_vals = find_confidence_interval(profile_chi_squared_ctg, ctg_values, min(profile_chi_squared_ctg), 1)
    
    frozen_ctg_label = add_val_label(frozen_ctg_vals)
    profile_ctg_label = add_val_label(profile_ctg_vals)

    
    # Plot results

    plt.figure(figsize=(16, 12))
    plt.suptitle(f"Frozen and Profile $\chi^2$ Scans ({order} Order)", fontsize=30)

    # c_g scan
    plt.subplot(2, 2, 1)
    plt.plot(cg_values, frozen_chi_squared_cg, label=f"Frozen $\\chi^2(c_g, c_{{tg}} = {fixed_ctg})$ {frozen_cg_label}")
    plt.plot(cg_values, profile_chi_squared_cg, label=f"Profile $\\chi^2(c_g$ min($c_{{tg}}$)) {profile_cg_label}")
    plt.axhline(1, color='red', linestyle='--', label="68% CL ($\\chi^2 = 1$)")
    plt.xlabel(r"Wilson coefficient $c_{g}$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.grid()

    # Minimized c_tg for c_g scan
    plt.subplot(2, 2, 2)
    plt.plot(cg_values, minimized_ctg_for_cg, label=r"Minimised $c_{tg}$ for each $c_{g}$")
    plt.xlabel(r"Wilson coefficient $c_{g}$")
    plt.ylabel(r"Minimised $c_{tg}$")
    plt.legend()
    plt.grid()

    # c_tg scan
    plt.subplot(2, 2, 3)
    plt.plot(ctg_values, frozen_chi_squared_ctg, label=f"Frozen $\\chi^2(c_{{tg}}, c_g = {fixed_cg})$ {frozen_ctg_label}")
    plt.plot(ctg_values, profile_chi_squared_ctg, label=f"Profile $\\chi^2(c_g$ min($c_{{tg}}$)) {profile_ctg_label}")
    plt.axhline(1, color='red', linestyle='--', label="68% CL ($\\chi^2 = 1$)")
    plt.xlabel(r"Wilson coefficient $c_{tg}$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.grid()


    # Minimized c_g for c_tg scan
    plt.subplot(2, 2, 4)
    plt.plot(ctg_values, minimized_cg_for_ctg, label=r"Minimised $c_{g}$ for each $c_{tg}$")
    plt.xlabel(r"Wilson coefficient $c_{tg}$")
    plt.ylabel(r"Minimised $c_{g}$")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()    

    '''
    return {
        "cg_values": cg_values,
        "ctg_values": ctg_values,
        "frozen_chi_squared_cg": frozen_chi_squared_cg,
        "profile_chi_squared_cg": profile_chi_squared_cg,
        "frozen_chi_squared_ctg": frozen_chi_squared_ctg,
        "profile_chi_squared_ctg": profile_chi_squared_ctg,
    }
    '''
        

def chi_squared_grid(
        optimized_mus,
        hessian_matrix,
        range_of_values,
        optimal_coefficients,
        quadratic_order=True
    ):
        
    order = "Quadratic" if quadratic_order else "First"
    
    optimal_cg, optimal_ctg = optimal_coefficients
    
    cg_values = range_of_values.copy()
    ctg_values = range_of_values.copy()
    # Initialize a 2D grid for chi-squared values
    chi_squared_grid = np.zeros((len(cg_values), len(ctg_values)))

    # Calculate chi-squared for each combination of c_g and c_tg
    for i, cg in enumerate(cg_values):
        for j, ctg in enumerate(ctg_values):
            delta_mu = optimized_mus - mu_c(cg, ctg, quadratic_order)
            chi_squared_grid[i, j] = delta_mu.T @ hessian_matrix @ delta_mu

    contour_levels = [2.3, 5.99]

    plt.figure(figsize=(10, 8))


    cg_grid, ctg_grid = np.meshgrid(cg_values, ctg_values)  # Create grid for plotting

    contour_plot = plt.contour(cg_grid, ctg_grid, chi_squared_grid.T, levels=contour_levels, colors=['yellow', 'green'], linestyles=['--', '-'])
    plt.clabel(contour_plot, fmt={2.3: '68%', 5.99: '95%'}, inline=True, fontsize=20)  # Add labels to the contours


    plt.contourf(cg_grid, ctg_grid, chi_squared_grid.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
    plt.colorbar(label=r"$\chi^2$")

    plt.scatter(optimal_cg, optimal_ctg, color='red', label='Minimum $\chi^2$', zorder=5)

    plt.xlabel(r"Wilson coefficient $c_{g}$")
    plt.ylabel(r"Wilson coefficient $c_{tg}$")
    plt.title(rf"Heatmap of $\chi^2(c_g, c_{{tg}})$ [{order} Order]")
    plt.legend(frameon=True, edgecolor='black', loc='best')
    plt.grid()
    plt.show()


def compute_chi2_hessian(cg, ctg, optimized_mus, NLL_hessian_matrix, quadratic_order, epsilon=1e-8):
    def chi2_func(c_g, c_tg):
        delta_mu = optimized_mus - mu_c(c_g, c_tg, quadratic_order)
        return float(delta_mu.T @ NLL_hessian_matrix @ delta_mu)
    # Numerical approximation of second derivatives
    d2_cg = (chi2_func(cg + epsilon, ctg) - 2 * chi2_func(cg, ctg) + chi2_func(cg - epsilon, ctg)) / (epsilon**2)
    d2_ctg = (chi2_func(cg, ctg + epsilon) - 2 * chi2_func(cg, ctg) + chi2_func(cg, ctg - epsilon)) / (epsilon**2)
    d2_cross = (chi2_func(cg + epsilon, ctg + epsilon) - chi2_func(cg + epsilon, ctg) - chi2_func(cg, ctg + epsilon) + chi2_func(cg, ctg)) / (epsilon**2)
    
    return np.array([[d2_cg, d2_cross],
                     [d2_cross, d2_ctg]])



