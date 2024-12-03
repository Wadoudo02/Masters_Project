#%%
import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from utils import *
from background_dist import *
from nll import *
from EFT import *
from selection import *
from categorisation import *

# Constants
total_lumi = 7.9804
target_lumi = 300
mass = 125

cats = {0: "0-60",
        1: "60-120",
        2: "120-200",
        3: "200-300",
        4: "300-"}

col_name = "_sel"

# Load dataframes
dfs = get_dfs(sample_path)

#Apply selection
dfs = get_selection(dfs)
#Categorisation
dfs = get_categorisation(dfs)

# Extract different cat integers
cats_unique = list(set(dfs["background"]['category']))

plot_diphoton_mass(dfs, cats_unique)

#%%
# Simple binned likelihood fit to mass histograms in signal window (120,130)
print('''
Building hists with background replacement
''')
hists = {}
mass_range = (120,130)
mass_bins = 5
v = 'mass'+col_name
for cat in cats_unique:
    hists[cat] = {}
    for proc in procs.keys():
        #Adding background events from background distribution
        if proc=="background":
            hist_counts=get_back_int(dfs["background"], cat, mass_range, mass_bins, len(cats_unique))
        else:
            cat_mask = dfs[proc]['category'] == cat
            hist_counts = np.histogram(dfs[proc][cat_mask][v], mass_bins, mass_range, weights=dfs[proc][cat_mask]['true_weight'+col_name])[0]
        hists[cat][proc] = hist_counts

NLL_vals = []
mu_vals = np.linspace(-2,5,100)

tth_new_samples = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
conf_matrix_raw, conf_matrix, conf_matrix_recon = get_conf_mat(tth_new_samples) #conf_matrix[2] is the one normalised by recon


#%%
'''
Scanning using combined hist
'''
print('''
NLL FROZEN SCAN USING COMBINED HISTOGRAM
''')
print(f'''Before combining
      {hists}
      ''')
comb_hist = build_combined_histogram(hists, conf_matrix, mass_bins=5)

plot_comb_hist(comb_hist)
plot_comb_hist(comb_hist, inc_background=False)

#%%
'''
Profiled fit for NLL 
'''
print('''
NLL PROFILED SCAN USING COMBINED HISTOGRAM
''')
best_mus = np.ones(5)
fig, ax = plt.subplots(1, 5, figsize=(40, 7))
for idx in range(len(conf_matrix)):
    best_mus[idx], nll_mu = profiled_NLL_fit(comb_hist,conf_matrix, idx, mu_vals)
    vals = find_crossings((mu_vals, TwoDeltaNLL(nll_mu)), 1.)
    print("Best mu for idx:", idx, "is", vals[0] ,"+/-", vals[1:])
    label = add_val_label(vals)
    ax[idx].plot(mu_vals, TwoDeltaNLL(nll_mu), label=label)
    ax[idx].set_title(f"NLL variation for mu idx: {idx}")
    ax[idx].set_ylabel("2 delta NLL")
    ax[idx].set_xlabel("mu")
    ax[idx].axvline(1., label="SM (expected)", color='green', alpha=0.5)
    ax[idx].axhline(1, color='grey', alpha=0.5, ls='--')
    ax[idx].axhline(4, color='grey', alpha=0.5, ls='--')
    ax[idx].set_ylim(0, 8)
    ax[idx].legend(loc='best')
fig.suptitle("Profiled NLL fit for each mu")
fig.savefig(f"{analysis_path}/nll_profiled_fit.png")
#%%
print('''
      NLL Global fit
''')

global_mu, global_nll, hessian, cov = global_nll_fit(comb_hist, conf_matrix)
print(f"""
      Global best fit for mu: {global_mu}
        Hessian: {hessian}
        Covariance: {cov}
""")

# %%
print('''
Getting Hessian matrix from combined hist
''')
fig, ax = plt.subplots(1, 3, figsize=(25, 5))
hessian_comb = np.zeros((len(conf_matrix), len(conf_matrix)))

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        hessian_comb[i][j] = get_hessian_comb(i,j, comb_hist, best_mus)
print("Hessian from comb hist: \n", hessian_comb)
show_matrix(hessian_comb, "Hessian matrix from comb hist", ax[0])
show_matrix(get_cov(hessian_comb), "Covariance matrix from comb hist", ax[1])
correlation_matrix = get_correlation_matrix(get_cov(hessian_comb))
show_matrix(correlation_matrix, "Correlation matrix", ax[2])
print(get_uncertainties(get_cov(hessian_comb)))
# %%
'''
Chi squared fit of mu(c)
'''
#print(c_g_coef, c_tg_coef)

c_vals = np.linspace(-10, 10, 1000)
c_tg=0
chi_squared = []
second_order = True

for c_g in c_vals:
    chi2 = get_chi_squared(best_mus, c_g, c_tg, hessian_comb, second_order=second_order)
    #print(chi2)
    chi_squared.append(chi2)

plt.figure(figsize=(8, 6))
plt.plot(c_vals, chi_squared, label=f"$\\chi^2(c_g, c_{{tg}} = {c_tg})$")
plt.axhline(2.3, color='red', linestyle='--', label='Threshold')
plt.xlabel(r"Wilson coefficient $c_{g}$")
plt.ylabel(r"$\chi^2(c_{g}, c_{tg})$")
plt.title("$\chi^2$ as a function of Wilson coefficient $c_{g}$")
plt.legend()
plt.grid()
plt.show()


# %%
'''
Chi Squared minimisation
'''
init_guess = 0
c_g, c_tg = 0,0

chi_2_c_g = []
chi_2_c_tg = []

for c_g in c_vals:
    res = minimize(lambda x: get_chi_squared(best_mus, c_g, x, hessian_comb,second_order=second_order),
                    init_guess, method='Nelder-Mead')
    chi_2_c_g.append(res.fun)

for c_tg in c_vals:
    res = minimize(lambda x: get_chi_squared(best_mus, x, c_tg, hessian_comb,second_order=second_order),
                    init_guess, method='Nelder-Mead')
    chi_2_c_tg.append(res.fun)

best_c_g = c_vals[np.argmin(chi_2_c_g)].round(2)
min_chi2_c_g = min(chi_2_c_g)
best_c_tg = c_vals[np.argmin(chi_2_c_tg)].round(2)
min_chi2_c_tg = min(chi_2_c_tg)

conf_interval_68_c_g = find_confidence_interval(chi_2_c_g, c_vals, min_chi2_c_g, 1)
conf_interval_68_c_tg = find_confidence_interval(chi_2_c_tg, c_vals, min_chi2_c_tg, 1)

print(f"Best fit for c_g: {best_c_g}")
print(f"68% confidence interval for c_g: {conf_interval_68_c_g}")
print(f"Best fit for c_tg: {best_c_tg}")
print(f"68% confidence interval for c_tg: {conf_interval_68_c_tg}")

# Plot the results
fig, ax=plt.subplots(1,2,figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(c_vals, chi_2_c_g, label=fr"Best fit: {best_c_g:.2f} $^{{+{conf_interval_68_c_g[1] - best_c_g:.2f}}}_{{-{best_c_g - conf_interval_68_c_g[0]:.2f}}}$")
plt.xlabel('c_g')
#Delta chi2 = 1 for 68% CI and 3.84 for 95% CI as technically only 1 degree of freedom here
plt.axhline(min_chi2_c_tg+1, color='red', linestyle='--', label='68% CI')
plt.axhline(min_chi2_c_tg+3.84, color='red', linestyle='--', label='95% CI')
plt.ylim(0,10)
plt.ylabel('Chi-squared')
plt.title('Profiled Scan over c_g')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(c_vals, chi_2_c_tg,label=fr"Best fit: {best_c_g:.2f} $^{{+{conf_interval_68_c_tg[1] - best_c_g:.2f}}}_{{-{best_c_g - conf_interval_68_c_tg[0]:.2f}}}$")
plt.xlabel('c_g')
plt.ylim(0,10)
plt.axhline(min_chi2_c_g+1, color='red', linestyle='--', label='68% CI')
plt.axhline(min_chi2_c_g+3.84, color='red', linestyle='--', label='95% CI')
plt.ylabel('Chi-squared')
plt.title('Profiled Scan over c_tg')
plt.legend()

plt.tight_layout()
plt.show()

# %%
'''
Grid minimisation
'''
width = 10

cg_values = np.linspace(-width//2, width//2, 100)  # Adjust range as needed
ctg_values = np.linspace(-width//2, width//2, 100)  # Adjust range as needed

# Initialize a 2D grid for chi-squared values
chi_squared_grid = np.zeros((len(cg_values), len(ctg_values)))  

for i, cg in enumerate(cg_values):
    for j, ctg in enumerate(ctg_values):
        chi_squared_grid[i][j] = get_chi_squared(best_mus, cg, ctg, hessian_comb, second_order=second_order)

plt.figure(figsize=(10, 8))


cg_grid, ctg_grid = np.meshgrid(cg_values, ctg_values)  # Create grid for plotting

contour_plot = plt.contour(cg_grid, ctg_grid, chi_squared_grid.T, levels=[2.3, 5.99], colors=['yellow', 'green'], linestyles=['--', '-'])
plt.clabel(contour_plot, fmt={2.3: '68%', 5.99: '95%'}, inline=True, fontsize=20)  # Add labels to the contours


plt.contourf(cg_grid, ctg_grid, chi_squared_grid.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
plt.colorbar(label=r"$\chi^2$")

plt.scatter(best_c_g, best_c_tg, color='red', label='Minimum $\chi^2$', zorder=5)

plt.xlabel(r"Wilson coefficient $c_{g}$")
plt.ylabel(r"Wilson coefficient $c_{tg}$")
plt.title(r"Heatmap of $\chi^2(c_g, c_{tg})$"+f" {'(second order)' if second_order else '(first order)'}")
plt.legend(frameon=True, edgecolor='black', loc='best')
plt.grid()
plt.show()

#%%
'''
Global Minimisation
'''
def objective_chi_sq(params):
    c_g, c_tg = params
    return get_chi_squared(global_mu, c_g, c_tg, hessian_comb, second_order=True)

initial_guess = [0, 0]
res = minimize(objective_chi_sq, initial_guess, method='BFGS')

# Extract the best fit values and the Hessian matrix
best_c_g, best_c_tg = res.x
cov_at_min = res.hess_inv
hessian_at_min = np.linalg.inv(cov_at_min)
correlation_at_min = get_correlation_matrix(cov_at_min)
uncs = get_uncertainties(cov_at_min)

fig, ax = plt.subplots(1, 3, figsize=(10, 5))

show_matrix(cov_at_min, "Covariance matrix at the minimum", ax[0])
show_matrix(hessian_at_min, "Hessian matrix at the minimum", ax[1])
show_matrix(correlation_at_min, "Correlation matrix at the minimum", ax[2])

print(f"Best fit values: c_g = {best_c_g}, c_tg = {best_c_tg}")
print(f"Hessian matrix at the minimum:\n{hessian_at_min}")
print(f"Correlation matrix at the minimum:\n{hessian_at_min}")


# %%
