import json


tth_json_path = "data/TTH.json"
with open(tth_json_path, "r") as file:
    wilson_data = json.load(file)

c_g_coef = wilson_data["data"]["central"]["a_cg"]
c_tg_coef = wilson_data["data"]["central"]["a_ctgre"]
order = [2, 5, 3, 4, 6]
c_g_coef_ordered = [c_g_coef[i] for i in order]
c_tg_coef_ordered = [c_tg_coef[i] for i in order]

def mu_c(c_g, c_tg, second_order=False):
    mus = []
    for i in range(len(c_g_coef_ordered)):
        curmu = float(1 + c_g*c_g_coef_ordered[i]
                          + c_tg*c_tg_coef_ordered[i])
        mus.append(curmu)
    return mus



def get_chi_squared(mu, c_g, c_tg, hessian):
#    print("input ", mu, c_g, c_tg)
    del_mu = mu - mu_c(c_g, c_tg)
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
