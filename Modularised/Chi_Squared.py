
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

ttH_coefficients_data = json.load(open("data/TTH.json", 'r'))

'''
NOTES ON INDEX IN JSON FILE

[2] = TTH_PTH_0_60 = 0 to 60
[3] = TTH_PTH_120_200 = 120 to 200
[4] = TTH_PTH_200_300 = 200 to 300
[5] = TTH_PTH_60_120 = 60 to 120
[6] = TTH_PTH_GT300 = >300

or

0-60 = [2]
60-120 = [5]
120-200 = [3]
200-300 = [4]
>300 = [6]


'''

optimized_mus = np.array([1,1,1,1,1])

a_cg = ttH_coefficients_data['data']["central"]["a_cg"]
a_ctgre = ttH_coefficients_data['data']["central"]["a_ctgre"]

b_cg_cg = ttH_coefficients_data['data']["central"]["b_cg_cg"]
b_cg_ctgre = ttH_coefficients_data['data']["central"]["b_cg_ctgre"]
b_ctgre_ctgre = ttH_coefficients_data['data']["central"]["b_ctgre_ctgre"]

# CHANGE TO C_G AND C_TG
def mu_c(cg, ctg, quadratic = True):
    """Theoretical model for mu as a function of Wilson coefficient c."""
    
    mu_0 = 1 + cg * a_cg[2] + ctg * a_ctgre[2]
    
    mu_1 = 1 + cg * a_cg[5] + ctg * a_ctgre[5]
    
    mu_2 = 1 + cg * a_cg[3] + ctg * a_ctgre[3]
    
    mu_3 = 1 + cg * a_cg[4] + ctg * a_ctgre[4]
    
    mu_4 = 1 + cg * a_cg[6] + ctg * a_ctgre[6]
    
    if quadratic:
        mu_0 += (cg ** 2) * b_cg_cg[2] + cg * ctg * b_cg_ctgre[2] + (ctg ** 2) * b_ctgre_ctgre[2]
        
        mu_1 += (cg ** 2) * b_cg_cg[5] + cg * ctg * b_cg_ctgre[5] + (ctg ** 2) * b_ctgre_ctgre[5]
        
        mu_2 += (cg ** 2) * b_cg_cg[3] + cg * ctg * b_cg_ctgre[3] + (ctg ** 2) * b_ctgre_ctgre[3]
        
        mu_3 += (cg ** 2) * b_cg_cg[4] + cg * ctg * b_cg_ctgre[4] + (ctg ** 2) * b_ctgre_ctgre[4]
        
        mu_4 += (cg ** 2) * b_cg_cg[6] + cg * ctg * b_cg_ctgre[6] + (ctg ** 2) * b_ctgre_ctgre[6]
        
    
    return np.array([mu_0 , mu_1,  mu_2,  mu_3,  mu_4])


def chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order):
    # Extract scalar values if arrays are passed
    cg_val = cg[0] if isinstance(cg, np.ndarray) else cg
    ctg_val = ctg[0] if isinstance(ctg, np.ndarray) else ctg
    
    delta_mu = optimized_mus - mu_c(cg_val, ctg_val, quadratic_order)
    return float(delta_mu.T @ hessian_matrix @ delta_mu)



def chi_squared_scans(
    optimized_mus,
    hessian_matrix,
    range_of_values,
    quadratic_order=True,
    fixed_ctg=0,
    fixed_cg=0,
):
    """
    Perform frozen and profile chi-squared scans for Wilson coefficients c_g and c_tg.

    Parameters:
        chi_squared_func (callable): Function to calculate chi-squared for given parameters.
        optimized_mus (array-like): Optimized mu values.
        hessian_matrix (array-like): Hessian matrix used in chi-squared calculation.
        range_of_values (array): Range of c_g values (min, max) for scanning.
        quadratic_order (bool): Whether to use quadratic or first-order approximation.
        plot (bool): Whether to plot the results.

    Returns:
        dict: Dictionary containing frozen and profile chi-squared values.
    """
    cg_values = range_of_values.copy()
    ctg_values = range_of_values.copy()

    profile_chi_squared_cg = []
    profile_chi_squared_ctg = []

    # Determine order description
    order = "Quadratic" if quadratic_order else "First"

    # Profile scans for c_g
    for cg in cg_values:
        result = minimize(lambda ctg: chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order), x0=0)
        profile_chi_squared_cg.append(result.fun)

    # Profile scans for c_tg
    for ctg in ctg_values:
        result = minimize(lambda cg: chi_squared_func(cg, ctg, optimized_mus, hessian_matrix, quadratic_order), x0=0)
        profile_chi_squared_ctg.append(result.fun)

    # Frozen scans
    frozen_chi_squared_cg = [chi_squared_func(cg, fixed_ctg, optimized_mus, hessian_matrix, quadratic_order) for cg in cg_values]
    frozen_chi_squared_ctg = [chi_squared_func(fixed_cg, ctg, optimized_mus, hessian_matrix, quadratic_order) for ctg in ctg_values]

    # Plot results
    
    plt.figure(figsize=(8, 12))
    plt.suptitle(f"Frozen and Profile $\chi^2$ Scans ({order} Order)", fontsize=30)

    # c_g scan
    plt.subplot(2, 1, 1)
    plt.plot(cg_values, frozen_chi_squared_cg, label=f"Frozen $\\chi^2(c_g, c_{{tg}} = {fixed_ctg})$")
    plt.plot(cg_values, profile_chi_squared_cg, label="Profile $\\chi^2(c_g)$ (minimized over $c_{tg}$)")
    plt.axhline(2.3, color='red', linestyle='--', label="68% CL ($\\chi^2 = 2.3$)")
    plt.xlabel(r"Wilson coefficient $c_{g}$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.grid()

    # c_tg scan
    plt.subplot(2, 1, 2)
    plt.plot(ctg_values, frozen_chi_squared_ctg, label=f"Frozen $\\chi^2(c_{{tg}}, c_g = {fixed_cg})$")
    plt.plot(ctg_values, profile_chi_squared_ctg, label="Profile $\\chi^2(c_{tg})$ (minimized over $c_{g}$)")
    plt.axhline(2.3, color='red', linestyle='--', label="68% CL ($\\chi^2 = 2.3$)")
    plt.xlabel(r"Wilson coefficient $c_{tg}$")
    plt.ylabel(r"$\chi^2$")
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