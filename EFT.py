import json
from utils import order_data


tth_json_path = "TTH.json"
with open(tth_json_path, "r") as file:
    wilson_data = json.load(file)

c_g_a_coef = wilson_data["data"]["central"]["a_cg"]
c_tg_a_coef = wilson_data["data"]["central"]["a_ctgre"]

c_g_sq_b_coef = wilson_data["data"]["central"]["b_cg_cg"]
c_g_c_tg_b_coef = wilson_data["data"]["central"]["b_cg_ctgre"]
c_tg_sq_b_coef = wilson_data["data"]["central"]["b_ctgre_ctgre"]

order = [2, 5, 3, 4, 6]
c_g_a_coef_ordered, c_tg_a_coef_ordered, c_g_sq_b_coef_ordered, c_g_c_tg_b_coef_ordered, c_tg_sq_b_coef_ordered = order_data(c_g_a_coef, c_tg_a_coef, c_g_sq_b_coef, c_g_c_tg_b_coef, c_tg_sq_b_coef, order=order)

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