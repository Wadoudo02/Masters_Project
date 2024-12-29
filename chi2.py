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


a_cg = wilson_data["data"]["central"]["a_cg"]
a_ctg = wilson_data["data"]["central"]["a_ctgre"]

b_cg_cg = wilson_data["data"]["central"]["b_cg_cg"]
b_cg_ctg = wilson_data["data"]["central"]["b_cg_ctgre"]
b_ctg_ctg = wilson_data["data"]["central"]["b_ctgre_ctgre"]
if standalone:
    order = [1,2,3,4,5]
else:
    order = [2, 5, 3, 4, 6]

(a_cg_ord, a_ctg_ord,
b_cg_cg_ord, b_cg_ctg_ord,
b_ctg_ctg_ord) = order_data(a_cg, a_ctg,
                                     b_cg_cg, b_cg_ctg,
                                     b_ctg_ctg, order=order)
def mu_c(c_g, c_tg, second_order=False):
    mus = []
    for i in range(len(a_cg_ord)):
        curmu = float(1 + c_g*a_cg_ord[i]
                          + c_tg*a_ctg_ord[i])
        if second_order:
            curmu += float((c_g**2)*b_cg_cg_ord[i]
                             + c_g*c_tg*b_cg_ctg_ord[i]
                             + (c_tg**2)*b_ctg_ctg_ord[i])

        mus.append(curmu)
    return mus



def get_chi_squared(mu, c_g, c_tg, hessian, second_order=False):
#    print("input ", mu, c_g, c_tg)
    del_mu = mu - mu_c(c_g, c_tg , second_order=second_order)
    #print("del_mu ", del_mu, "cg ", c_g, "ctg", c_tg)
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