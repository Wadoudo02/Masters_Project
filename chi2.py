#%%
import json
from utils import order_data
import numpy as np

tth_standalone_json_path = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass2/TTH_standalone.json"
tth_json_path = "TTH.json"
standalone = True

path = tth_standalone_json_path if standalone else tth_json_path

with open(path, "r") as file:
    wilson_data = json.load(file)


c_g_a_coef = wilson_data["data"]["central"]["a_cg"]
c_tg_a_coef = wilson_data["data"]["central"]["a_ctgre"]

c_g_sq_b_coef = wilson_data["data"]["central"]["b_cg_cg"]
c_g_c_tg_b_coef = wilson_data["data"]["central"]["b_cg_ctgre"]
c_tg_sq_b_coef = wilson_data["data"]["central"]["b_ctgre_ctgre"]

if standalone:
    order = [1,2,3,4,5]
else:
    order = [2, 5, 3, 4, 6]

(c_g_a_coef_ordered, c_tg_a_coef_ordered,
c_g_sq_b_coef_ordered, c_g_c_tg_b_coef_ordered,
c_tg_sq_b_coef_ordered) = order_data(c_g_a_coef, c_tg_a_coef,
                                     c_g_sq_b_coef, c_g_c_tg_b_coef,
                                     c_tg_sq_b_coef, order=order)

def mu_c(c_g, c_tg, second_order=False):
    mus = []
    for i in range(len(c_g_a_coef_ordered)):
        curmu = float(1 + c_g*c_g_a_coef_ordered[i]
                          + c_tg*c_tg_a_coef_ordered[i])
        if second_order:
            curmu += float((c_g**2)*c_g_sq_b_coef_ordered[i]
                             + c_g*c_tg*c_g_c_tg_b_coef_ordered[i]
                             + (c_tg**2)*c_tg_sq_b_coef_ordered[i])

        mus.append(curmu)
    return mus



def get_chi_squared(mu, c_g, c_tg, hessian, second_order=False):
#    print("input ", mu, c_g, c_tg)
    del_mu = mu - mu_c(c_g, c_tg , second_order=second_order)
    chi2= del_mu.T @ hessian @ del_mu
    return chi2

def find_confidence_interval(chi_2, c_vals, min_chi_2, delta_chi_2):
    lower_bound = None
    upper_bound = None
    for i, chi in enumerate(chi_2):
        if chi <= min_chi_2 + delta_chi_2:
            if lower_bound is None:
                lower_bound = c_vals[i]
            upper_bound = c_vals[i]
    return lower_bound, upper_bound

def get_chi_squared_hessian(mu, c_g, c_tg, hessian, second_order=False, eps=1e-5):
    # Initialize the Hessian matrix
    hess_chi2 = np.zeros((2, 2))

    # Compute chi-squared for different perturbations
    chi2_00 = get_chi_squared(mu, c_g, c_tg, hessian, second_order=second_order)  # f(c_g, c_{tg})
    chi2_10 = get_chi_squared(mu, c_g + eps, c_tg, hessian, second_order=second_order)  # f(c_g + eps, c_{tg})
    chi2_01 = get_chi_squared(mu, c_g, c_tg + eps, hessian, second_order=second_order)  # f(c_g, c_{tg} + eps)
    chi2_11 = get_chi_squared(mu, c_g + eps, c_tg + eps, hessian, second_order=second_order)  # f(c_g + eps, c_{tg} + eps)

    # Compute second derivatives using central differences
    hess_chi2[0, 0] = (chi2_10 - 2 * chi2_00 + get_chi_squared(mu, c_g - eps, c_tg, hessian, second_order=second_order)) / (eps**2)  # ∂²/∂c_g²
    hess_chi2[1, 1] = (chi2_01 - 2 * chi2_00 + get_chi_squared(mu, c_g, c_tg - eps, hessian, second_order=second_order)) / (eps**2)  # ∂²/∂c_{tg}²
    hess_chi2[0, 1] = (chi2_11 - chi2_10 - chi2_01 + chi2_00) / (eps**2)  # ∂²/∂c_g∂c_{tg}
    hess_chi2[1, 0] = hess_chi2[0, 1]  # Hessian is symmetric

    return hess_chi2